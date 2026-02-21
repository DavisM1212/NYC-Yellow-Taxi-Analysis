import calendar
import json
import pathlib
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "taxi_parquets"
ARTIFACT_ROOT_DIR = PROJECT_ROOT / "cluster_outputs"
VALID_YEARS = set(range(2015, 2023))
DEFAULT_SAMPLE_COLUMNS = [
    "fare_amount",
    "tolls_amount",
    "passenger_count",
    "trip_distance",
    "duration_min",
    "speed_mph",
    "tip_rate",
    "is_airport_trip",
    "pickup_hour",
    "dow",
    "month_num",
    "weekend",
    "rush_hour",
    "night_trip",
    "pickup_borough",
    "pickup_zone_name",
    "dropoff_borough",
    "dropoff_zone_name"
]
DEFAULT_BEHAVIORAL_COLUMNS = [
    "fare_amount",
    "tolls_amount",
    "passenger_count",
    "trip_distance",
    "duration_min",
    "speed_mph",
    "tip_rate_pct",
    "is_airport_trip_num"
]
DEFAULT_WINSOR_COLUMNS = [
    "fare_amount",
    "tolls_amount",
    "trip_distance",
    "duration_min",
    "speed_mph",
    "tip_rate_pct"
]
HOUR_BIN_EDGES = [-1, 3, 6, 15, 20, 24]
HOUR_BIN_LABELS = ["Late Night", "Airport Rush", "Daytime", "Evening Commute", "Nightlife"]
CLUSTER_REPORT_FILE = "cluster_report.md"
CLUSTER_MANIFEST_FILE = "cluster_manifest.json"
CLUSTER_PERSONA_FILE = "cluster_personas.md"
CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS = 250_000
DEFAULT_K_SELECT_METHOD = "consensus"
DEFAULT_AUTO_MIN_K = 3
DEFAULT_MIN_CLUSTER_PCT = 2.0
CONSENSUS_WEIGHTS = {
    "silhouette": 0.40,
    "elbow_drop_pct": 0.30,
    "calinski_harabasz": 0.20,
    "cluster_balance": 0.10
}
DEFAULT_GRID_CLUSTER_ALGO = "auto"
DEFAULT_FINAL_CLUSTER_ALGO = "auto"
DEFAULT_MINIBATCH_THRESHOLD_ROWS = 750_000
DEFAULT_MINIBATCH_BATCH_SIZE = 20_000
DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT = 20
DEFAULT_MINIBATCH_REASSIGNMENT_RATIO = 0.01
DEFAULT_CALINSKI_SAMPLE_SIZE = 200_000


def _validate_year(year):
    if year not in VALID_YEARS:
        raise ValueError(f"year must be in 2015..2022 (got {year})")


def _parquet_path_for_year(year):
    return (PARQUET_DIR / f"yellow_clean_{year}.parquet").as_posix()


def _connect_duckdb():
    con = duckdb.connect()
    con.execute("SET enable_object_cache = true;")
    con.execute("SET threads = 4;")
    con.execute("SET memory_limit='10GB';")
    return con


def _resolve_cluster_algo(cluster_algo, n_rows, minibatch_threshold_rows):
    if cluster_algo not in {"auto", "kmeans", "minibatch"}:
        raise ValueError(f"cluster_algo must be auto, kmeans, or minibatch (got {cluster_algo})")
    if cluster_algo == "auto":
        return "minibatch" if int(n_rows) >= int(minibatch_threshold_rows) else "kmeans"
    return cluster_algo


# -----------------------------------------------------------------------------
# Fetch reproducible sample for clustering
# -----------------------------------------------------------------------------
def fetch_cluster_year(year, target_rows=1_000_000, seed=42, columns=None, verbose=True):
    _validate_year(year)
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive (got {target_rows})")

    selected_cols = list(columns) if columns is not None else DEFAULT_SAMPLE_COLUMNS
    parquet_path = _parquet_path_for_year(year)
    select_clause = ",\n      ".join(selected_cols)

    con = _connect_duckdb()
    try:
        total_rows = con.execute(f"SELECT count(*) FROM read_parquet('{parquet_path}')").fetchone()
        total_rows = total_rows[0] if total_rows is not None else 0

        # Reservoir sampling keeps memory stable and stays reproducible via REPEATABLE
        query = f"""
        PRAGMA disable_progress_bar;

        SELECT
          {select_clause}
        FROM read_parquet('{parquet_path}')
        USING SAMPLE reservoir({int(target_rows)} ROWS) REPEATABLE ({int(seed)})
        """

        if verbose:
            print(f"[{year}] Creating clustering sample...")
        df = con.execute(query).fetch_df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"[{year}] Sampling returned 0 rows. Check parquet path: {parquet_path}")

    if verbose:
        sampled_pct = (len(df) / float(total_rows)) * 100.0 if total_rows else 0.0
        print(
            f"[{year}] Requested {target_rows:,} rows (seed={seed}), "
            f"received {len(df):,} rows ({sampled_pct:.3f}% of {total_rows:,})"
        )

    string_cols = ["pickup_borough", "pickup_zone_name", "dropoff_borough", "dropoff_zone_name"]
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype("category")

    bool_cols = ["is_airport_trip", "weekend", "rush_hour", "night_trip"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8).astype(bool)

    numeric_cols = [
        "fare_amount",
        "tolls_amount",
        "passenger_count",
        "trip_distance",
        "duration_min",
        "speed_mph",
        "tip_rate",
        "pickup_hour",
        "dow",
        "month_num"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------------------------------------------------------
# Feature preparation for clustering and descriptors
# -----------------------------------------------------------------------------
def prepare_cluster_frame(df, verbose=True):
    use_df = df.copy()

    if "tip_rate" not in use_df.columns:
        raise KeyError("tip_rate column is required for clustering")

    use_df["tip_rate_pct"] = pd.to_numeric(use_df["tip_rate"], errors="coerce") * 100.0

    if "pickup_hour" in use_df.columns:
        use_df["hour_bin"] = pd.cut(
            pd.to_numeric(use_df["pickup_hour"], errors="coerce"),
            bins=HOUR_BIN_EDGES,
            labels=HOUR_BIN_LABELS,
            right=False
        )
        use_df["hour_bin"] = use_df["hour_bin"].astype("category")
    else:
        use_df["hour_bin"] = pd.Categorical(["Unknown"] * len(use_df))

    if "dow" in use_df.columns:
        use_df["dow_name"] = pd.to_numeric(use_df["dow"], errors="coerce").map(
            lambda x: calendar.day_name[int(x) - 1] if pd.notna(x) and 1 <= int(x) <= 7 else "Unknown"
        )
    else:
        use_df["dow_name"] = "Unknown"
    use_df["dow_name"] = use_df["dow_name"].astype("category")

    if "month_num" in use_df.columns:
        use_df["month_name"] = pd.to_numeric(use_df["month_num"], errors="coerce").map(
            lambda x: calendar.month_name[int(x)] if pd.notna(x) and 1 <= int(x) <= 12 else "Unknown"
        )
    else:
        use_df["month_name"] = "Unknown"
    use_df["month_name"] = use_df["month_name"].astype("category")

    # One-hot fields are used for descriptive profiling, not for behavioral cluster fit
    dummy_df = pd.get_dummies(
        use_df,
        prefix=["hour", "day", "month"],
        columns=["hour_bin", "dow_name", "month_name"],
        dtype=np.int8
    )

    for c in ["is_airport_trip", "weekend", "rush_hour", "night_trip"]:
        if c in dummy_df.columns:
            dummy_df[c] = dummy_df[c].astype(np.int8)

    if "is_airport_trip" not in dummy_df.columns:
        dummy_df["is_airport_trip"] = 0

    dummy_df["is_airport_trip_num"] = pd.to_numeric(dummy_df["is_airport_trip"], errors="coerce").fillna(0).astype(float)

    descriptive_cols = []
    descriptive_cols += [c for c in dummy_df.columns if c.startswith("hour_")]
    descriptive_cols += [c for c in dummy_df.columns if c.startswith("day_")]
    descriptive_cols += [c for c in dummy_df.columns if c.startswith("month_") and c != "month_num"]
    descriptive_cols += [c for c in ["weekend", "rush_hour", "night_trip"] if c in dummy_df.columns]

    if verbose:
        print(f"Prepared frame rows: {len(dummy_df):,}")
        print(f"Behavioral columns available: {len([c for c in DEFAULT_BEHAVIORAL_COLUMNS if c in dummy_df.columns])}")
        print(f"Descriptive columns available: {len(descriptive_cols)}")

    return dummy_df, descriptive_cols


# -----------------------------------------------------------------------------
# Robust winsorization and matrix preparation
# -----------------------------------------------------------------------------
def fit_winsor_bounds(df, cols, lower=0.003, upper=0.997):
    if not (0 <= lower < upper <= 1):
        raise ValueError(f"lower/upper must satisfy 0 <= lower < upper <= 1 (got {lower}, {upper})")

    bounds = {}
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        bounds[c] = (float(lo), float(hi))
        rows.append(
            {
                "column": c,
                "lower_quantile": float(lower),
                "upper_quantile": float(upper),
                "lower_bound": float(lo),
                "upper_bound": float(hi)
            }
        )
    return bounds, pd.DataFrame(rows)


def apply_winsor_bounds(df, bounds):
    out = df.copy()
    clip_counts = {}
    for c, (lo, hi) in bounds.items():
        s = pd.to_numeric(out[c], errors="coerce")
        clipped = s.clip(lower=lo, upper=hi)
        clip_counts[c] = int((s != clipped).sum())
        out[c] = clipped
    clip_df = pd.DataFrame({"column": list(clip_counts.keys()), "cells_clipped": list(clip_counts.values())})
    return out, clip_df


def build_behavioral_matrix(df, behavioral_cols=DEFAULT_BEHAVIORAL_COLUMNS, winsor_cols=DEFAULT_WINSOR_COLUMNS, lower=0.003, upper=0.997, verbose=True):
    missing_behavioral = [c for c in behavioral_cols if c not in df.columns]
    if len(missing_behavioral) > 0:
        raise KeyError(f"Missing required behavioral columns: {missing_behavioral}")

    X = df[behavioral_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    winsor_cols_in = [c for c in winsor_cols if c in X.columns]
    bounds, bounds_df = fit_winsor_bounds(X, winsor_cols_in, lower=lower, upper=upper)
    X_winsor, clip_df = apply_winsor_bounds(X, bounds)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_winsor)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_winsor.columns, index=X_winsor.index)

    if verbose:
        print(f"Behavioral matrix rows: {len(X_winsor):,} | cols: {X_winsor.shape[1]}")
        print(f"Winsor columns used: {winsor_cols_in}")
        if len(clip_df) > 0:
            total_clipped = int(clip_df["cells_clipped"].sum())
            print(f"Total clipped cells: {total_clipped:,}")

    preprocess_summary = {
        "winsor_lower": float(lower),
        "winsor_upper": float(upper),
        "winsor_cols_used": winsor_cols_in,
        "n_rows": int(len(X_winsor)),
        "n_features": int(X_winsor.shape[1]),
        "total_cells_clipped": int(clip_df["cells_clipped"].sum()) if len(clip_df) > 0 else 0
    }

    return X_winsor, X_scaled_df, scaler, bounds_df, clip_df, preprocess_summary


# -----------------------------------------------------------------------------
# K selection diagnostics
# -----------------------------------------------------------------------------
def _silhouette_on_sample(X_scaled, labels, sample_size=100_000, seed=42):
    n = len(X_scaled)
    if n < 3:
        return np.nan
    if sample_size is None or n <= sample_size:
        return float(silhouette_score(X_scaled, labels))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(sample_size), replace=False)
    return float(silhouette_score(X_scaled[idx], labels[idx]))


def _calinski_on_sample(X_scaled, labels, sample_size=DEFAULT_CALINSKI_SAMPLE_SIZE, seed=42):
    n = len(X_scaled)
    if n < 3:
        return np.nan
    if sample_size is None or n <= sample_size:
        return float(calinski_harabasz_score(X_scaled, labels))

    rng = np.random.default_rng(seed + 17)
    idx = rng.choice(n, size=int(sample_size), replace=False)
    return float(calinski_harabasz_score(X_scaled[idx], labels[idx]))


def evaluate_kmeans_grid(
        X_scaled_df,
        k_min=2,
        k_max=12,
        seed=42,
        n_init=20,
        max_iter=300,
        silhouette_sample_size=100_000,
        calinski_sample_size=DEFAULT_CALINSKI_SAMPLE_SIZE,
        cluster_algo=DEFAULT_GRID_CLUSTER_ALGO,
        minibatch_threshold_rows=DEFAULT_MINIBATCH_THRESHOLD_ROWS,
        minibatch_batch_size=DEFAULT_MINIBATCH_BATCH_SIZE,
        minibatch_max_no_improvement=DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT,
        minibatch_reassignment_ratio=DEFAULT_MINIBATCH_REASSIGNMENT_RATIO,
        verbose=True):
    if k_min < 2:
        raise ValueError(f"k_min must be >=2 (got {k_min})")
    if k_max < k_min:
        raise ValueError(f"k_max must be >= k_min (got {k_max} < {k_min})")

    X = X_scaled_df.to_numpy()
    n_rows = len(X)
    algo_used = _resolve_cluster_algo(cluster_algo, n_rows=n_rows, minibatch_threshold_rows=minibatch_threshold_rows)
    rows = []

    for k in range(k_min, k_max + 1):
        start = time.perf_counter()
        if algo_used == "minibatch":
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                n_init=n_init,
                max_iter=max_iter,
                batch_size=int(min(minibatch_batch_size, n_rows)),
                max_no_improvement=int(minibatch_max_no_improvement),
                reassignment_ratio=float(minibatch_reassignment_ratio)
            )
        else:
            km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        labels = km.fit_predict(X)
        cluster_frac = pd.Series(labels).value_counts(normalize=True)
        inertia = float(km.inertia_)
        sil = _silhouette_on_sample(X, labels, sample_size=silhouette_sample_size, seed=seed)
        ch = _calinski_on_sample(X, labels, sample_size=calinski_sample_size, seed=seed)
        runtime = float(time.perf_counter() - start)
        rows.append(
            {
                "k": int(k),
                "inertia": inertia,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "min_cluster_pct": float(cluster_frac.min() * 100.0),
                "max_cluster_pct": float(cluster_frac.max() * 100.0),
                "cluster_algo_used": algo_used,
                "fit_seconds": runtime
            }
        )
        if verbose:
            print(
                f"k={k:2d} | algo={algo_used} | inertia={inertia:,.1f} | "
                f"silhouette={sil:.4f} | calinski_harabasz={ch:,.1f} | sec={runtime:.2f}"
            )

    metrics_df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    metrics_df["inertia_drop"] = metrics_df["inertia"].shift(1) - metrics_df["inertia"]
    metrics_df["inertia_drop_pct"] = metrics_df["inertia_drop"] / metrics_df["inertia"].shift(1)
    return metrics_df


def _minmax_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.nan, index=s.index)
    if abs(float(s_max) - float(s_min)) < 1e-12:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def select_k(metrics_df, method=DEFAULT_K_SELECT_METHOD, min_k_for_auto=DEFAULT_AUTO_MIN_K, min_cluster_pct=DEFAULT_MIN_CLUSTER_PCT):
    if method not in {"silhouette", "calinski_harabasz", "consensus"}:
        raise ValueError(f"method must be silhouette, calinski_harabasz, or consensus (got {method})")

    use_df = metrics_df.copy()
    if min_k_for_auto is not None:
        cand_df = use_df[use_df["k"] >= int(min_k_for_auto)].copy()
        if len(cand_df) > 0:
            use_df = cand_df

    if "min_cluster_pct" in use_df.columns and min_cluster_pct is not None:
        bal_df = use_df[use_df["min_cluster_pct"] >= float(min_cluster_pct)].copy()
        if len(bal_df) > 0:
            use_df = bal_df

    if method in {"silhouette", "calinski_harabasz"}:
        metric_col = "silhouette" if method == "silhouette" else "calinski_harabasz"
        use_df = use_df.dropna(subset=[metric_col])
    else:
        use_df = use_df.dropna(subset=["silhouette", "calinski_harabasz"])

    if use_df.empty:
        raise RuntimeError("Could not select k because candidate metrics are unavailable after filtering")

    if method == "consensus":
        score_df = use_df.copy()
        # Normalize each metric onto [0, 1] before weighted consensus aggregation
        score_df["score_silhouette"] = _minmax_scale(score_df["silhouette"]).fillna(0.0)
        score_df["score_calinski_harabasz"] = _minmax_scale(score_df["calinski_harabasz"]).fillna(0.0)
        score_df["score_elbow_drop_pct"] = _minmax_scale(score_df["inertia_drop_pct"].fillna(0.0)).fillna(0.0)
        score_df["score_cluster_balance"] = _minmax_scale(score_df["min_cluster_pct"]).fillna(0.0)
        score_df["consensus_score"] = (
            CONSENSUS_WEIGHTS["silhouette"] * score_df["score_silhouette"]
            + CONSENSUS_WEIGHTS["elbow_drop_pct"] * score_df["score_elbow_drop_pct"]
            + CONSENSUS_WEIGHTS["calinski_harabasz"] * score_df["score_calinski_harabasz"]
            + CONSENSUS_WEIGHTS["cluster_balance"] * score_df["score_cluster_balance"]
        )
        metric_col = "consensus_score"
        best_idx = score_df[metric_col].idxmax()
        best_row = score_df.loc[best_idx]
        return int(best_row["k"]), metric_col, float(best_row[metric_col]), score_df

    best_idx = use_df[metric_col].idxmax() # type: ignore
    best_row = use_df.loc[best_idx]
    return int(best_row["k"]), metric_col, float(best_row[metric_col]), use_df # type: ignore


def plot_k_selection(metrics_df, out_path=None, show_plot=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="steelblue")
    axes[0].set_title("Elbow: Inertia by k")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Silhouette and Calinski-Harabasz are on different scales, so use twin y-axes
    ax_left = axes[1]
    ax_right = ax_left.twinx()
    ax_left.plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="seagreen", label="Silhouette")
    ax_right.plot(metrics_df["k"], metrics_df["calinski_harabasz"], marker="o", color="darkorange", label="Calinski-Harabasz")
    ax_left.set_title("Cluster Quality by k")
    ax_left.set_xlabel("k")
    ax_left.set_ylabel("Silhouette", color="seagreen")
    ax_right.set_ylabel("Calinski-Harabasz", color="darkorange")
    ax_left.tick_params(axis="y", labelcolor="seagreen")
    ax_right.tick_params(axis="y", labelcolor="darkorange")
    ax_left.grid(True, linestyle="--", alpha=0.4)
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Clustering fit and profile tables
# -----------------------------------------------------------------------------
def fit_behavioral_clusters(
        X_scaled_df,
        k,
        seed=42,
        n_init=30,
        max_iter=500,
        cluster_algo=DEFAULT_FINAL_CLUSTER_ALGO,
        minibatch_threshold_rows=DEFAULT_MINIBATCH_THRESHOLD_ROWS,
        minibatch_batch_size=DEFAULT_MINIBATCH_BATCH_SIZE,
        minibatch_max_no_improvement=DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT,
        minibatch_reassignment_ratio=DEFAULT_MINIBATCH_REASSIGNMENT_RATIO):
    X = X_scaled_df.to_numpy()
    n_rows = len(X)
    algo_used = _resolve_cluster_algo(cluster_algo, n_rows=n_rows, minibatch_threshold_rows=minibatch_threshold_rows)

    if algo_used == "minibatch":
        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=int(min(minibatch_batch_size, n_rows)),
            max_no_improvement=int(minibatch_max_no_improvement),
            reassignment_ratio=float(minibatch_reassignment_ratio)
        )
    else:
        model = KMeans(n_clusters=int(k), random_state=seed, n_init=n_init, max_iter=max_iter)

    labels = model.fit_predict(X)
    return model, pd.Series(labels, index=X_scaled_df.index, name="cluster"), algo_used


def _build_cluster_size_table(df):
    size = df["cluster"].value_counts().sort_index().rename("n_rows").reset_index()
    size = size.rename(columns={"index": "cluster"})
    size["cluster"] = size["cluster"].astype(int)
    size["n_rows"] = size["n_rows"].astype(int)
    size["pct_rows"] = (size["n_rows"] / size["n_rows"].sum()) * 100.0
    return size


def _build_behavioral_summary(df, behavioral_cols):
    means = df.groupby("cluster")[behavioral_cols].mean().round(4)
    overall = df[behavioral_cols].mean()
    rel_index = means.divide(overall, axis=1).round(3)
    return means, rel_index


def _build_descriptive_summary(df, descriptive_cols):
    if len(descriptive_cols) == 0:
        return pd.DataFrame(index=sorted(df["cluster"].unique()))
    return (df.groupby("cluster")[descriptive_cols].mean() * 100.0).round(2)


def _build_category_mix(df, category_col):
    mix = (
        df.groupby("cluster", observed=False)[category_col]
        .value_counts(normalize=True)
        .mul(100.0)
        .rename("pct")
        .reset_index()
    )
    return mix.pivot(index="cluster", columns=category_col, values="pct").fillna(0.0).round(2)


def _build_top_zones(df, zone_col, top_n=10):
    cluster_totals = df.groupby("cluster").size().rename("cluster_total")
    top = (
        df.groupby(["cluster", zone_col], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    top["rank"] = top.groupby("cluster")["count"].rank(method="first", ascending=False).astype(int)
    top = top[top["rank"] <= int(top_n)].copy()
    top = top.merge(cluster_totals, on="cluster", how="left")
    # pct_within_cluster is share of all cluster trips, pct_top_n_share is share within the displayed top_n only
    top["pct_within_cluster"] = (top["count"] / top["cluster_total"]) * 100.0
    top["pct_top_n_share"] = (
        top["count"] / top.groupby("cluster")["count"].transform("sum")
    ) * 100.0
    top = top.sort_values(["cluster", "rank"]).reset_index(drop=True)

    wide = top.pivot(index="cluster", columns="rank", values=zone_col)
    wide = wide.rename(columns=lambda r: f"Top {r}")
    return top, wide


def build_cluster_summaries(cluster_df, behavioral_cols, descriptive_cols, top_n=10):
    size_table = _build_cluster_size_table(cluster_df)
    behavioral_means, behavioral_index = _build_behavioral_summary(cluster_df, behavioral_cols)
    descriptive_rates = _build_descriptive_summary(cluster_df, descriptive_cols)
    pickup_borough_mix = _build_category_mix(cluster_df, "pickup_borough")
    dropoff_borough_mix = _build_category_mix(cluster_df, "dropoff_borough")
    top_pickup_long, top_pickup_wide = _build_top_zones(cluster_df, "pickup_zone_name", top_n=top_n)
    top_dropoff_long, top_dropoff_wide = _build_top_zones(cluster_df, "dropoff_zone_name", top_n=top_n)

    return {
        "cluster_size": size_table,
        "behavioral_means": behavioral_means,
        "behavioral_index": behavioral_index,
        "descriptive_rates_pct": descriptive_rates,
        "pickup_borough_mix_pct": pickup_borough_mix,
        "dropoff_borough_mix_pct": dropoff_borough_mix,
        "top_pickup_zones_long": top_pickup_long,
        "top_pickup_zones_wide": top_pickup_wide,
        "top_dropoff_zones_long": top_dropoff_long,
        "top_dropoff_zones_wide": top_dropoff_wide
    }


def _persona_label(behavior_row, overall_tip_rate):
    airport_share = float(behavior_row.get("is_airport_trip_num", 0.0))
    pax = float(behavior_row.get("passenger_count", np.nan))
    dist = float(behavior_row.get("trip_distance", np.nan))
    dur = float(behavior_row.get("duration_min", np.nan))
    tip = float(behavior_row.get("tip_rate_pct", np.nan))

    if airport_share >= 0.40 and dist >= 8:
        return "Airport Connector Long-Haul"
    if pax >= 2.5:
        return "Group-Cab City Movers"
    if dist <= 2.0 and dur <= 10 and tip >= overall_tip_rate:
        return "High-Tipping Local Errands"
    if tip < overall_tip_rate and dist >= 3.0:
        return "Lower-Tip Mid-Range Trips"
    return "Core Manhattan Mixed Trips"


def _persona_comment(label, tip_rank, n_clusters):
    if tip_rank == 1:
        rank_msg = "Highest tip propensity"
    elif tip_rank == n_clusters:
        rank_msg = "Lowest tip propensity"
    else:
        rank_msg = "Mid-tier tip propensity"

    if "Airport" in label:
        return f"{rank_msg}, airport-driven"
    if "Group-Cab" in label:
        return f"{rank_msg}, group-passenger segment"
    if "High-Tipping" in label:
        return f"{rank_msg}, short-hop urban core"
    if "Lower-Tip" in label:
        return f"{rank_msg}, mixed-value longer hauls"
    return rank_msg


def _top_pct_lines(row, prefix, top_n=3):
    cols = [c for c in row.index if c.startswith(prefix)]
    if len(cols) == 0:
        return []
    s = row[cols].sort_values(ascending=False).head(top_n)
    return [f"{c.replace(prefix, '')}: {float(v):.2f}%" for c, v in s.items()]


def _top_mix_lines(mix_df, cluster, top_n=2):
    if cluster not in mix_df.index:
        return []
    s = pd.to_numeric(mix_df.loc[cluster], errors="coerce").sort_values(ascending=False).head(top_n)
    return [f"{str(k)}: {float(v):.2f}%" for k, v in s.items()]


def _top_zones_for_cluster(top_zone_long_df, cluster, zone_col, top_n=5):
    use = top_zone_long_df[top_zone_long_df["cluster"] == cluster].sort_values("rank").head(top_n)
    if zone_col not in use.columns:
        return []
    return use[zone_col].astype(str).tolist()


def _persona_interpretation(label):
    if label == "Airport Connector Long-Haul":
        return "Airport-to-city and city-to-airport trips dominate this cluster, with high fares and long distances driving steady but not top tip percentages"
    if label == "Group-Cab City Movers":
        return "Higher passenger counts suggest shared trips, tourism, and social travel patterns with moderate fare levels and generally solid tipping"
    if label == "High-Tipping Local Errands":
        return "Short, dense urban hops with strong tipping norms and high transaction frequency make this cluster especially attractive despite smaller fares"
    if label == "Lower-Tip Mid-Range Trips":
        return "Mid-range city hauls show weaker tip behavior relative to fare and time, indicating lower conversion into generous tipping"
    return "Broad city usage profile with mixed trip purposes and balanced tipping behavior"


def _persona_actions(label):
    if label == "Airport Connector Long-Haul":
        return "Stage near JFK/LGA corridors and Manhattan gateways, prioritize queue positioning and clean airport transitions"
    if label == "Group-Cab City Movers":
        return "Target nightlife, hotel, and entertainment pickup zones where multi-passenger demand is common"
    if label == "High-Tipping Local Errands":
        return "Prioritize dense Manhattan short-hop zones during daytime and evening windows, optimizing turnover rate"
    if label == "Lower-Tip Mid-Range Trips":
        return "Use this cluster for volume fill rather than tip optimization, and focus on service quality nudges to improve tip conversion"
    return "Maintain broad coverage and shift toward higher-tip zones during peak opportunity windows"


def build_cluster_persona_overview(summary_tables):
    behavior = summary_tables["behavioral_means"].copy()
    size = summary_tables["cluster_size"].set_index("cluster").copy()
    overall_tip = float(behavior["tip_rate_pct"].mean())
    rank = behavior["tip_rate_pct"].rank(ascending=False, method="dense").astype(int)
    n_clusters = int(behavior.index.nunique())

    rows = []
    for cluster, row in behavior.iterrows():
        label = _persona_label(row, overall_tip_rate=overall_tip)
        tip_rank = int(rank.loc[cluster])
        rows.append(
            {
                "cluster": int(cluster),
                "persona": label,
                "tip_rate_pct": float(row["tip_rate_pct"]),
                "tip_rank": tip_rank,
                "cluster_share_pct": float(size.loc[cluster, "pct_rows"]) if cluster in size.index else np.nan,
                "comment": _persona_comment(label, tip_rank=tip_rank, n_clusters=n_clusters)
            }
        )

    return pd.DataFrame(rows).sort_values(["tip_rank", "cluster"]).reset_index(drop=True)


def build_cluster_personas_markdown(year, run_config, summary_tables, persona_overview):
    behavior = summary_tables["behavioral_means"]
    behavior_idx = summary_tables["behavioral_index"]
    desc = summary_tables["descriptive_rates_pct"]
    pu_mix = summary_tables["pickup_borough_mix_pct"]
    do_mix = summary_tables["dropoff_borough_mix_pct"]
    top_pu_long = summary_tables["top_pickup_zones_long"]
    top_do_long = summary_tables["top_dropoff_zones_long"]

    lines = [
        f"# NYC Yellow Taxi Cluster Personas ({year})",
        "",
        f"Sample rows: {run_config['sample_rows']:,}",
        "",
        "## Tip-Ranked Persona Overview",
        _df_to_markdown_table(
            persona_overview[["cluster", "persona", "tip_rate_pct", "tip_rank", "cluster_share_pct", "comment"]],
            float_fmt=".2f"
        ),
        ""
    ]

    for _, prow in persona_overview.sort_values("tip_rank").iterrows():
        cluster = int(prow["cluster"])
        label = str(prow["persona"])
        b = behavior.loc[cluster]
        bi = behavior_idx.loc[cluster]
        d = desc.loc[cluster] if cluster in desc.index else pd.Series(dtype=float)

        top_hours = _top_pct_lines(d, "hour_", top_n=3)
        top_days = _top_pct_lines(d, "day_", top_n=2)
        top_months = _top_pct_lines(d, "month_", top_n=3)
        pu_boro = _top_mix_lines(pu_mix, cluster, top_n=2)
        do_boro = _top_mix_lines(do_mix, cluster, top_n=2)
        pu_zones = _top_zones_for_cluster(top_pu_long, cluster, "pickup_zone_name", top_n=5)
        do_zones = _top_zones_for_cluster(top_do_long, cluster, "dropoff_zone_name", top_n=5)

        lines += [
            f"## Cluster {cluster} - {label}",
            "**Behavioral signature**",
            f"- Fare: ${float(b['fare_amount']):.2f} ({float(bi['fare_amount']):.2f}x overall)",
            f"- Distance/Duration/Speed: {float(b['trip_distance']):.2f} miles, {float(b['duration_min']):.2f} min, {float(b['speed_mph']):.2f} mph",
            f"- Passenger count: {float(b['passenger_count']):.2f}",
            f"- Airport share: {float(b['is_airport_trip_num']) * 100.0:.2f}%",
            f"- Tip rate: {float(b['tip_rate_pct']):.2f}%",
            "",
            "**Time pattern**",
            f"- Top hour bins: {', '.join(top_hours) if len(top_hours) > 0 else 'N/A'}",
            f"- Top day mix: {', '.join(top_days) if len(top_days) > 0 else 'N/A'}",
            f"- Top month mix: {', '.join(top_months) if len(top_months) > 0 else 'N/A'}",
            "",
            "**Spatial footprint**",
            f"- Pickup borough mix (top): {', '.join(pu_boro) if len(pu_boro) > 0 else 'N/A'}",
            f"- Dropoff borough mix (top): {', '.join(do_boro) if len(do_boro) > 0 else 'N/A'}",
            f"- Top pickup zones: {', '.join(pu_zones) if len(pu_zones) > 0 else 'N/A'}",
            f"- Top dropoff zones: {', '.join(do_zones) if len(do_zones) > 0 else 'N/A'}",
            "",
            "**Interpretation**",
            f"- {_persona_interpretation(label)}",
            "",
            "**Actionable driver insight**",
            f"- {_persona_actions(label)}",
            ""
        ]

    return "\n".join(lines)


def _df_to_markdown_table(df, float_fmt=".2f"):
    header = "| " + " | ".join([str(c) for c in df.columns]) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for row in df.to_dict(orient="records"):
        vals = []
        for val in row.values():
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (int, np.integer)):
                vals.append(str(int(val)))
            elif isinstance(val, (float, np.floating)):
                vals.append(format(float(val), float_fmt))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, divider] + rows)


def build_cluster_report_markdown(year, run_config, preprocess_summary, k_metric_col, k_metric_val, summary_tables):
    size = summary_tables["cluster_size"].copy()
    top_behavior = summary_tables["behavioral_means"].copy()

    top_tip = top_behavior["tip_rate_pct"].sort_values(ascending=False).head(2)
    top_tip_lines = [f"- Cluster {int(idx)}: {val:.2f}% tip_rate" for idx, val in top_tip.items()]

    lines = [
        f"# NYC Yellow Taxi Behavioral Clustering Report ({year})",
        "",
        "## Scope",
        f"- Sample rows: {run_config['sample_rows']:,}",
        f"- Behavioral feature count: {run_config['behavioral_feature_count']}",
        f"- Selected k: {run_config['selected_k']}",
        f"- k selection metric: `{k_metric_col}` = {k_metric_val:.4f}",
        f"- k selection method: `{run_config['k_select_method']}`",
        f"- Grid clustering algorithm used: `{run_config['grid_cluster_algo_used']}`",
        f"- Final clustering algorithm used: `{run_config['final_cluster_algo_used']}`",
        f"- Winsor bounds: {preprocess_summary['winsor_lower']} to {preprocess_summary['winsor_upper']}",
        "",
        "## Cluster Sizes",
        _df_to_markdown_table(size),
        "",
        "## Highest Tip-Rate Clusters",
        *top_tip_lines,
        "",
        "## Notes",
        "- Clustering is unsupervised and descriptive",
        "- `tip_rate_pct` is included as a behavioral monetary signal by design",
        "- Cluster patterns are predictive segments, not causal proof"
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Export artifacts
# -----------------------------------------------------------------------------
def _sample_assignment_export(df, max_rows=CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS, seed=42):
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def export_cluster_artifacts(year, run_config, preprocess_summary, k_grid_metrics, k_metric_col, k_metric_val, cluster_df, summaries, bounds_df, clip_df, scaler, model, behavioral_cols=DEFAULT_BEHAVIORAL_COLUMNS, export_assignment_sample=True, assignment_max_rows=CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS, out_dir=ARTIFACT_ROOT_DIR, write_report=True, write_manifest=True, show_plot=False):
    year_dir = pathlib.Path(out_dir, f"year_{year}")
    year_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {}
    persona_overview = build_cluster_persona_overview(summaries)

    centroid_scaled = pd.DataFrame(model.cluster_centers_, columns=list(behavioral_cols))
    centroid_scaled.insert(0, "cluster", np.arange(len(centroid_scaled)))
    centroid_raw_vals = scaler.inverse_transform(model.cluster_centers_)
    centroid_raw = pd.DataFrame(centroid_raw_vals, columns=list(behavioral_cols))
    centroid_raw.insert(0, "cluster", np.arange(len(centroid_raw)))

    export_frames = {
        "k_selection_metrics": k_grid_metrics,
        "cluster_size_summary": summaries["cluster_size"],
        "cluster_behavioral_means": summaries["behavioral_means"].reset_index(),
        "cluster_behavioral_index": summaries["behavioral_index"].reset_index(),
        "cluster_persona_overview": persona_overview,
        "kmeans_centroids_scaled": centroid_scaled,
        "kmeans_centroids_raw_units": centroid_raw,
        "cluster_descriptive_rates_pct": summaries["descriptive_rates_pct"].reset_index(),
        "cluster_pickup_borough_mix_pct": summaries["pickup_borough_mix_pct"].reset_index(),
        "cluster_dropoff_borough_mix_pct": summaries["dropoff_borough_mix_pct"].reset_index(),
        "cluster_top_pickup_zones_long": summaries["top_pickup_zones_long"],
        "cluster_top_pickup_zones_wide": summaries["top_pickup_zones_wide"].reset_index(),
        "cluster_top_dropoff_zones_long": summaries["top_dropoff_zones_long"],
        "cluster_top_dropoff_zones_wide": summaries["top_dropoff_zones_wide"].reset_index(),
        "winsor_bounds": bounds_df,
        "winsor_clip_counts": clip_df,
        "scaler_feature_summary": pd.DataFrame(
            {
                "feature": list(behavioral_cols),
                "mean_": list(scaler.mean_),
                "scale_": list(scaler.scale_)
            }
        )
    }

    if export_assignment_sample:
        keep_cols = [
            "cluster",
            "fare_amount",
            "tolls_amount",
            "passenger_count",
            "trip_distance",
            "duration_min",
            "speed_mph",
            "tip_rate_pct",
            "is_airport_trip",
            "pickup_borough",
            "dropoff_borough",
            "pickup_zone_name",
            "dropoff_zone_name"
        ]
        keep_cols = [c for c in keep_cols if c in cluster_df.columns]
        assignment_sample = _sample_assignment_export(cluster_df[keep_cols], max_rows=assignment_max_rows, seed=run_config["seed"])
        export_frames["cluster_assignment_sample"] = assignment_sample

    for name, frame in export_frames.items():
        out_path = year_dir / f"{name}.csv"
        frame.to_csv(out_path.as_posix(), index=False)
        csv_paths[name] = out_path.as_posix()

    k_plot_file = year_dir / "k_selection_plot.png"
    plot_k_selection(k_grid_metrics, out_path=k_plot_file.as_posix(), show_plot=show_plot)
    plot_path = k_plot_file.as_posix()

    report_path = None
    personas_path = None
    if write_report:
        report_text = build_cluster_report_markdown(
            year=year,
            run_config=run_config,
            preprocess_summary=preprocess_summary,
            k_metric_col=k_metric_col,
            k_metric_val=k_metric_val,
            summary_tables=summaries
        )
        report_path = (year_dir / CLUSTER_REPORT_FILE).as_posix()
        pathlib.Path(report_path).write_text(report_text, encoding="utf-8")

        personas_text = build_cluster_personas_markdown(
            year=year,
            run_config=run_config,
            summary_tables=summaries,
            persona_overview=persona_overview
        )
        personas_path = (year_dir / CLUSTER_PERSONA_FILE).as_posix()
        pathlib.Path(personas_path).write_text(personas_text, encoding="utf-8")

    manifest_path = None
    if write_manifest:
        manifest = {
            "year": int(year),
            "run_config": run_config,
            "preprocess_summary": preprocess_summary,
            "selected_k_metric": {"name": k_metric_col, "value": float(k_metric_val)},
            "k_selection_plot": plot_path,
            "csv_paths": csv_paths,
            "report_path": report_path,
            "personas_path": personas_path,
            "export_assignment_sample": bool(export_assignment_sample),
            "assignment_max_rows": int(assignment_max_rows)
        }
        manifest_path = (year_dir / CLUSTER_MANIFEST_FILE).as_posix()
        pathlib.Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "year_dir": year_dir.as_posix(),
        "csv_paths": csv_paths,
        "plot_path": plot_path,
        "report_path": report_path,
        "personas_path": personas_path,
        "manifest_path": manifest_path
    }


# -----------------------------------------------------------------------------
# End-to-end yearly clustering pipeline
# -----------------------------------------------------------------------------
def run_behavioral_clustering_year(
        year,
        target_rows=1_000_000,
        seed=42,
        selected_k=None,
        k_min=2,
        k_max=12,
        k_select_method=DEFAULT_K_SELECT_METHOD,
        silhouette_sample_size=100_000,
        calinski_sample_size=DEFAULT_CALINSKI_SAMPLE_SIZE,
        min_k_for_auto=DEFAULT_AUTO_MIN_K,
        min_cluster_pct=DEFAULT_MIN_CLUSTER_PCT,
        grid_cluster_algo=DEFAULT_GRID_CLUSTER_ALGO,
        final_cluster_algo=DEFAULT_FINAL_CLUSTER_ALGO,
        minibatch_threshold_rows=DEFAULT_MINIBATCH_THRESHOLD_ROWS,
        minibatch_batch_size=DEFAULT_MINIBATCH_BATCH_SIZE,
        minibatch_max_no_improvement=DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT,
        minibatch_reassignment_ratio=DEFAULT_MINIBATCH_REASSIGNMENT_RATIO,
        winsor_lower=0.003,
        winsor_upper=0.997,
        top_n_zones=10,
        export_assignment_sample=True,
        assignment_max_rows=CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS,
        out_dir=ARTIFACT_ROOT_DIR,
        write_report=True,
        write_manifest=True,
        show_plot=False,
        verbose=True):
    _validate_year(year)
    if selected_k is not None and selected_k < 2:
        raise ValueError(f"selected_k must be >=2 (got {selected_k})")

    raw_df = fetch_cluster_year(
        year=year,
        target_rows=target_rows,
        seed=seed,
        columns=DEFAULT_SAMPLE_COLUMNS,
        verbose=verbose
    )

    prepared_df, descriptive_cols = prepare_cluster_frame(raw_df, verbose=verbose)
    X_raw, X_scaled_df, scaler, bounds_df, clip_df, preprocess_summary = build_behavioral_matrix(
        prepared_df,
        behavioral_cols=DEFAULT_BEHAVIORAL_COLUMNS,
        winsor_cols=DEFAULT_WINSOR_COLUMNS,
        lower=winsor_lower,
        upper=winsor_upper,
        verbose=verbose
    )

    k_grid_metrics = evaluate_kmeans_grid(
        X_scaled_df,
        k_min=k_min,
        k_max=k_max,
        seed=seed,
        n_init=20,
        max_iter=300,
        silhouette_sample_size=silhouette_sample_size,
        calinski_sample_size=calinski_sample_size,
        cluster_algo=grid_cluster_algo,
        minibatch_threshold_rows=minibatch_threshold_rows,
        minibatch_batch_size=minibatch_batch_size,
        minibatch_max_no_improvement=minibatch_max_no_improvement,
        minibatch_reassignment_ratio=minibatch_reassignment_ratio,
        verbose=verbose
    )

    if selected_k is None:
        # Auto mode applies post-grid filtering and metric-driven selection
        use_k, metric_col, metric_val, scored_grid = select_k(
            k_grid_metrics,
            method=k_select_method,
            min_k_for_auto=min_k_for_auto,
            min_cluster_pct=min_cluster_pct
        )
        if metric_col not in k_grid_metrics.columns:
            k_grid_metrics = k_grid_metrics.merge(
                scored_grid[["k", metric_col]].drop_duplicates(subset=["k"]),
                on="k",
                how="left"
            )
    else:
        use_k = int(selected_k)
        metric_col = "manual_k"
        metric_val = float(use_k)

    if verbose:
        print(f"Selected k: {use_k} using {metric_col}={metric_val:.4f}")

    model, labels, final_algo_used = fit_behavioral_clusters(
        X_scaled_df,
        k=use_k,
        seed=seed,
        n_init=30,
        max_iter=500,
        cluster_algo=final_cluster_algo,
        minibatch_threshold_rows=minibatch_threshold_rows,
        minibatch_batch_size=minibatch_batch_size,
        minibatch_max_no_improvement=minibatch_max_no_improvement,
        minibatch_reassignment_ratio=minibatch_reassignment_ratio
    )

    cluster_df = prepared_df.copy()
    # Keep summary tables in original units for direct interpretation and reporting
    cluster_df[DEFAULT_BEHAVIORAL_COLUMNS] = X_raw[DEFAULT_BEHAVIORAL_COLUMNS]
    cluster_df["cluster"] = labels.values

    summaries = build_cluster_summaries(
        cluster_df,
        behavioral_cols=DEFAULT_BEHAVIORAL_COLUMNS,
        descriptive_cols=descriptive_cols,
        top_n=top_n_zones
    )

    run_config = {
        "year": int(year),
        "seed": int(seed),
        "target_rows": int(target_rows),
        "sample_rows": int(len(cluster_df)),
        "selected_k": int(use_k),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "k_select_method": k_select_method,
        "min_k_for_auto": int(min_k_for_auto) if min_k_for_auto is not None else None,
        "min_cluster_pct": float(min_cluster_pct) if min_cluster_pct is not None else None,
        "silhouette_sample_size": int(silhouette_sample_size) if silhouette_sample_size is not None else None,
        "calinski_sample_size": int(calinski_sample_size) if calinski_sample_size is not None else None,
        "grid_cluster_algo": grid_cluster_algo,
        "grid_cluster_algo_used": (
            str(k_grid_metrics["cluster_algo_used"].iloc[0])
            if "cluster_algo_used" in k_grid_metrics.columns and len(k_grid_metrics) > 0
            else "unknown"
        ),
        "final_cluster_algo": final_cluster_algo,
        "final_cluster_algo_used": final_algo_used,
        "minibatch_threshold_rows": int(minibatch_threshold_rows),
        "minibatch_batch_size": int(minibatch_batch_size),
        "minibatch_max_no_improvement": int(minibatch_max_no_improvement),
        "minibatch_reassignment_ratio": float(minibatch_reassignment_ratio),
        "winsor_lower": float(winsor_lower),
        "winsor_upper": float(winsor_upper),
        "top_n_zones": int(top_n_zones),
        "behavioral_feature_count": int(len(DEFAULT_BEHAVIORAL_COLUMNS)),
        "descriptive_feature_count": int(len(descriptive_cols)),
        "kmeans_inertia": float(model.inertia_),
        "export_assignment_sample": bool(export_assignment_sample),
        "assignment_max_rows": int(assignment_max_rows)
    }

    artifact_result = export_cluster_artifacts(
        year=year,
        run_config=run_config,
        preprocess_summary=preprocess_summary,
        k_grid_metrics=k_grid_metrics,
        k_metric_col=metric_col,
        k_metric_val=metric_val,
        cluster_df=cluster_df,
        summaries=summaries,
        bounds_df=bounds_df,
        clip_df=clip_df,
        scaler=scaler,
        model=model,
        behavioral_cols=DEFAULT_BEHAVIORAL_COLUMNS,
        export_assignment_sample=export_assignment_sample,
        assignment_max_rows=assignment_max_rows,
        out_dir=out_dir,
        write_report=write_report,
        write_manifest=write_manifest,
        show_plot=show_plot
    )

    summary_row = {
        "year": int(year),
        "sample_rows": int(len(cluster_df)),
        "selected_k": int(use_k),
        "k_select_metric": metric_col,
        "k_select_metric_value": float(metric_val),
        "inertia": float(model.inertia_),
        "cluster_count": int(cluster_df["cluster"].nunique()),
        "top_cluster_tip_rate_pct": float(
            summaries["behavioral_means"]["tip_rate_pct"].max()
        ),
        "bottom_cluster_tip_rate_pct": float(
            summaries["behavioral_means"]["tip_rate_pct"].min()
        )
    }

    return {
        "model": model,
        "cluster_df": cluster_df,
        "k_grid_metrics": k_grid_metrics,
        "summary_tables": summaries,
        "summary_row": summary_row,
        "artifact_result": artifact_result
    }


def run_behavioral_clustering_exports(
        years=VALID_YEARS,
        target_rows=1_000_000,
        seed=42,
        selected_k=None,
        k_min=2,
        k_max=12,
        k_select_method=DEFAULT_K_SELECT_METHOD,
        silhouette_sample_size=100_000,
        calinski_sample_size=DEFAULT_CALINSKI_SAMPLE_SIZE,
        min_k_for_auto=DEFAULT_AUTO_MIN_K,
        min_cluster_pct=DEFAULT_MIN_CLUSTER_PCT,
        grid_cluster_algo=DEFAULT_GRID_CLUSTER_ALGO,
        final_cluster_algo=DEFAULT_FINAL_CLUSTER_ALGO,
        minibatch_threshold_rows=DEFAULT_MINIBATCH_THRESHOLD_ROWS,
        minibatch_batch_size=DEFAULT_MINIBATCH_BATCH_SIZE,
        minibatch_max_no_improvement=DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT,
        minibatch_reassignment_ratio=DEFAULT_MINIBATCH_REASSIGNMENT_RATIO,
        winsor_lower=0.003,
        winsor_upper=0.997,
        top_n_zones=10,
        export_assignment_sample=True,
        assignment_max_rows=CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS,
        out_dir=ARTIFACT_ROOT_DIR,
        write_report=True,
        write_manifest=True,
        show_plot=False,
        verbose=True):
    summary_rows = []
    for year in sorted(years):
        if verbose:
            print("\n" + "=" * 72)
            print(f"Running behavioral clustering export for year {year}")
            print("=" * 72)
        result = run_behavioral_clustering_year(
            year=year,
            target_rows=target_rows,
            seed=seed,
            selected_k=selected_k,
            k_min=k_min,
            k_max=k_max,
            k_select_method=k_select_method,
            silhouette_sample_size=silhouette_sample_size,
            calinski_sample_size=calinski_sample_size,
            min_k_for_auto=min_k_for_auto,
            min_cluster_pct=min_cluster_pct,
            grid_cluster_algo=grid_cluster_algo,
            final_cluster_algo=final_cluster_algo,
            minibatch_threshold_rows=minibatch_threshold_rows,
            minibatch_batch_size=minibatch_batch_size,
            minibatch_max_no_improvement=minibatch_max_no_improvement,
            minibatch_reassignment_ratio=minibatch_reassignment_ratio,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            top_n_zones=top_n_zones,
            export_assignment_sample=export_assignment_sample,
            assignment_max_rows=assignment_max_rows,
            out_dir=out_dir,
            write_report=write_report,
            write_manifest=write_manifest,
            show_plot=show_plot,
            verbose=verbose
        )
        summary_rows.append(result["summary_row"])

    summary_df = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    out_path = pathlib.Path(out_dir, "all_years_cluster_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path.as_posix(), index=False)
    if verbose:
        print(f"\nSaved multi-year clustering summary -> {out_path.as_posix()}")
    return summary_df


if __name__ == "__main__":
    run_behavioral_clustering_year(
        year=2015,
        target_rows=5_000_000,
        seed=42,
        selected_k=4,
        k_min=2,
        k_max=10,
        k_select_method=DEFAULT_K_SELECT_METHOD,
        silhouette_sample_size=80_000,
        calinski_sample_size=DEFAULT_CALINSKI_SAMPLE_SIZE,
        min_k_for_auto=DEFAULT_AUTO_MIN_K,
        min_cluster_pct=DEFAULT_MIN_CLUSTER_PCT,
        grid_cluster_algo=DEFAULT_GRID_CLUSTER_ALGO,
        final_cluster_algo=DEFAULT_FINAL_CLUSTER_ALGO,
        minibatch_threshold_rows=DEFAULT_MINIBATCH_THRESHOLD_ROWS,
        minibatch_batch_size=DEFAULT_MINIBATCH_BATCH_SIZE,
        minibatch_max_no_improvement=DEFAULT_MINIBATCH_MAX_NO_IMPROVEMENT,
        minibatch_reassignment_ratio=DEFAULT_MINIBATCH_REASSIGNMENT_RATIO,
        winsor_lower=0.003,
        winsor_upper=0.997,
        top_n_zones=10,
        export_assignment_sample=True,
        assignment_max_rows=CLUSTER_ASSIGNMENT_EXPORT_MAX_ROWS,
        out_dir=ARTIFACT_ROOT_DIR,
        write_report=True,
        write_manifest=True,
        show_plot=False,
        verbose=True
    )
