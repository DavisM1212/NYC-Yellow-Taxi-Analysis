import calendar
import json
import pathlib
import sys
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from lightgbm import LGBMRegressor, early_stopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_ROOT / "taxi_parquets"
VALID_YEARS = set(range(2015, 2023))
VALID_TARGETS = {"tip_amount", "tip_rate", "log_tip"}
DUMMY_PREFIXES = ("hour_", "month_", "borough_")
LEAKAGE_DROP_COLS = {"tip_amount", "tip_rate", "log_tip", "total_amount"}
DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "n_estimators": 500,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
DEFAULT_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "criterion": "gini"
}
ARTIFACT_ROOT_DIR = PROJECT_ROOT / "model_outputs"
REG_PLOT_MAX_POINTS = 20_000
DEFAULT_REG_PLOT_CROP_QUANTILES = (0.01, 0.99)
DEFAULT_REG_PLOT_CROP_PADDING_FRAC = 0.03
MANIFEST_FILE_NAME = "run_manifest.json"
YEAR_REPORT_FILE_NAME = "year_report.md"

# Columns available in yellow_clean_YYYY.parquet after cleaning
SAMPLE_COLUMNS = [
    "passenger_count",
    "rate_code",
    "pickup_borough",
    "pickup_hour",
    "month_num",
    "dow",
    "weekend",
    "is_airport_trip",
    "rush_hour",
    "night_trip",
    "trip_distance",
    "duration_min",
    "speed_mph",
    "fare_amount",
    "extra",
    "mta_tax",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
    "tolls_amount",
    "tip_amount",
    "tip_rate",
    "total_amount"
]

# Keep target out of outlier cleaning to avoid target truncation bias
DEFAULT_OUTLIER_COLS = (
    "trip_distance",
    "duration_min",
    "speed_mph",
    "fare_amount",
    "tolls_amount",
    "extra"
)

DEFAULT_PROFILE_COLS = (
    "trip_distance",
    "duration_min",
    "speed_mph",
    "fare_amount",
    "tip_amount",
    "tip_rate"
)


def _validate_year(year):
    if year not in VALID_YEARS:
        raise ValueError(f"year must be in 2015..2022 (got {year})")


def _validate_target(target):
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {sorted(VALID_TARGETS)} (got {target})")


def _parquet_path_for_year(year):
    return (PARQUET_DIR / f"yellow_clean_{year}.parquet").as_posix()


def _connect_duckdb():
    con = duckdb.connect()
    con.execute("SET enable_object_cache = true;")
    con.execute("SET threads = 4;")
    con.execute("SET memory_limit='10GB';")
    return con


# -----------------------------------------------------------------------------
# Fetch a reproducible row sample for a given year
# -----------------------------------------------------------------------------
def fetch_regr_year(year, target_rows=500_000, seed=42, columns=None, verbose=True):
    _validate_year(year)
    if target_rows <= 0:
        raise ValueError(f"target_rows must be positive (got {target_rows})")

    selected_cols = list(columns) if columns is not None else SAMPLE_COLUMNS
    parquet_path = _parquet_path_for_year(year)
    select_clause = ",\n      ".join(selected_cols)

    con = _connect_duckdb()
    try:
        total_rows = con.execute(f"SELECT count(*) FROM read_parquet('{parquet_path}')").fetchone()
        total_rows = total_rows[0] if total_rows is not None else 0

        query = f"""
        PRAGMA disable_progress_bar;

        SELECT
          {select_clause}
        FROM read_parquet('{parquet_path}')
        USING SAMPLE reservoir({int(target_rows)} ROWS) REPEATABLE ({int(seed)})
        """

        if verbose:
            print(f"[{year}] Creating regression sample...")
        df = con.execute(query).fetch_df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError(f"[{year}] Sampling returned 0 rows. Check parquet path: {parquet_path}")

    if verbose:
        sampled_pct = (len(df) / float(total_rows)) * 100.0 if total_rows else 0.0
        print(f"[{year}] Requested {target_rows:,} rows (seed={seed}), "
              f"received {len(df):,} rows ({sampled_pct:.3f}% of {total_rows:,}).")

    if "pickup_borough" in df.columns:
        df["pickup_borough"] = df["pickup_borough"].fillna("Unknown").astype("category")

    zero_fill_cols = [
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "tolls_amount"
    ]
    for c in zero_fill_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    bool_cols = ["is_airport_trip", "weekend", "rush_hour", "night_trip"]
    for c in bool_cols:
        if c in df.columns:
            # Avoid NaN -> True casting behavior
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8).astype(bool)

    return df


# -----------------------------------------------------------------------------
# Print compact QA summary to verify sampling and schema assumptions
# -----------------------------------------------------------------------------
def print_data_health(df, title="Data Health Snapshot"):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Rows: {len(df):,}")
    print(f"Cols: {df.shape[1]:,}")

    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("Missingness: no nulls")
    else:
        print("Top missingness (%):")
        print((missing.head(10) * 100).round(3).to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary = df[numeric_cols].describe(percentiles=[0.01, 0.5, 0.99]).T
        keep = ["mean", "std", "min", "1%", "50%", "99%", "max"]
        print("\nNumeric summary (selected stats):")
        print(summary[keep].round(3).to_string())


# -----------------------------------------------------------------------------
# Get summary stats and diagnostic plots for one numeric column
# -----------------------------------------------------------------------------
def full_variable_check(df, column, bins=60, show_plots=True):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in DataFrame.")

    data = pd.to_numeric(df[column], errors="coerce").dropna()
    if data.empty:
        raise ValueError(f"Column '{column}' has no numeric values after coercion.")

    desc = data.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q1, q3 = desc["25%"], desc["75%"]
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = ((data < lower) | (data > upper)).sum()
    outlier_pct = (outlier_count / len(data)) * 100

    print(f"\nVariable: {column}")
    print(desc.to_string())
    print(f"Potential outliers beyond 1.5*IQR: {outlier_count:,} "
          f"({outlier_pct:.2f}% of {len(data):,} records)")
    print(f"Suggested clip range: [{lower:.3f}, {upper:.3f}]")

    if show_plots:
        fig, axes = plt.subplots(
            4, 1, figsize=(9, 14), gridspec_kw={"height_ratios": [3, 0.6, 3, 3]}
        )
        plt.subplots_adjust(hspace=0.6)

        sns.histplot(data, kde=True, bins=bins, color="steelblue", ax=axes[0])
        axes[0].set_title(f"Distribution of {column}", fontsize=14)
        axes[0].set_xlabel(column)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        sns.boxplot(x=data, color="lightcoral", ax=axes[1])
        axes[1].set_title(f"Boxplot of {column}")
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

        positive_data = data[data > 0]
        if len(positive_data) > 0:
            sns.histplot(
                positive_data,
                kde=True,
                bins=bins,
                log_scale=True,
                color="seagreen",
                ax=axes[2]
            )
            axes[2].set_title(f"Log-Scale Distribution of {column}", fontsize=14)
            axes[2].set_xlabel(f"log({column})")
            axes[2].grid(True, linestyle="--", alpha=0.4)
        else:
            axes[2].text(
                0.5,
                0.5,
                "No positive values for log-scale plot",
                ha="center",
                va="center",
                fontsize=12,
                color="gray"
            )
            axes[2].axis("off")

        stats.probplot(data, dist="norm", plot=axes[3])
        if len(axes[3].get_lines()) > 1:
            axes[3].get_lines()[1].set_color("red")
        axes[3].set_title(f"Normal Probability Plot (Q-Q) for {column}", fontsize=14)
        axes[3].grid(True, linestyle="--", alpha=0.4)
        plt.show()

    return {
        "column": column,
        "n": int(len(data)),
        "q01": float(desc["1%"]),
        "q99": float(desc["99%"]),
        "iqr_low": float(lower),
        "iqr_high": float(upper),
        "outlier_count_1p5_iqr": int(outlier_count),
        "outlier_pct_1p5_iqr": float(outlier_pct)
    }


# -----------------------------------------------------------------------------
# Clean outliers using configurable strategy
# method options: "drop", "winsor"
# -----------------------------------------------------------------------------
def _validate_outlier_inputs(df, cols, lower=0.003, upper=0.997, method="winsor"):
    if not (0 < lower < upper < 1):
        raise ValueError(f"Expected 0 < lower < upper < 1, got {lower=} {upper=}")
    if method not in {"drop", "winsor"}:
        raise ValueError(f"method must be one of ['drop', 'winsor'] (got {method})")

    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Outlier columns not in DataFrame: {missing}")

    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Outlier columns must be numeric: {non_numeric}")


def fit_outlier_bounds(df, cols, lower=0.003, upper=0.997, verbose=True):
    _validate_outlier_inputs(df, cols, lower=lower, upper=upper, method="winsor")

    work_df = df.copy()
    bounds = {}
    for c in cols:
        q_low, q_high = work_df[c].quantile([lower, upper])
        bounds[c] = (float(q_low), float(q_high))
        if verbose:
            before_min, before_max = work_df[c].min(), work_df[c].max()
            print(f"{c:20s}: bounds [{q_low:.3f}, {q_high:.3f}] "
                  f"(before [{before_min:.3f}, {before_max:.3f}])")
    return bounds


def apply_outlier_bounds(df, bounds, method="winsor", verbose=True):
    if method not in {"drop", "winsor"}:
        raise ValueError(f"method must be one of ['drop', 'winsor'] (got {method})")

    work_df = df.copy()
    cols = list(bounds.keys())
    missing = [c for c in cols if c not in work_df.columns]
    if missing:
        raise KeyError(f"Bound columns not in DataFrame: {missing}")

    if method == "winsor":
        clipped_df = work_df.copy()
        cells_clipped = 0
        for c, (low, high) in bounds.items():
            below = (clipped_df[c] < low).sum()
            above = (clipped_df[c] > high).sum()
            cells_clipped += int(below + above)
            clipped_df[c] = clipped_df[c].clip(lower=low, upper=high)

        summary = {
            "method": method,
            "rows_before": len(work_df),
            "rows_after": len(clipped_df),
            "rows_removed": 0,
            "rows_removed_pct": 0.0,
            "cells_clipped": int(cells_clipped)
        }
        if verbose:
            print(f"\nWinsorized {cells_clipped:,} cell values across {len(cols)} columns.")
        keep_mask = np.ones(len(work_df), dtype=bool)
        return clipped_df.reset_index(drop=True), summary, keep_mask

    # method == "drop"
    mask = np.ones(len(work_df), dtype=bool)
    for c, (low, high) in bounds.items():
        mask &= work_df[c].between(low, high, inclusive="both")

    n_before = len(work_df)
    filtered_df = work_df.loc[mask].reset_index(drop=True)
    n_after = len(filtered_df)
    removed = n_before - n_after
    removed_pct = (removed / n_before) * 100 if n_before else 0.0

    summary = {
        "method": method,
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_removed": int(removed),
        "rows_removed_pct": float(removed_pct),
        "cells_clipped": 0
    }
    if verbose:
        print(f"\nRemoved {removed:,} rows ({removed_pct:.2f}% of sample).")
    return filtered_df, summary, mask


def apply_outlier_bounds_to_xy(X, y, bounds, method="winsor", verbose=True, label="set"):
    X_clean, summary, keep_mask = apply_outlier_bounds(X, bounds, method=method, verbose=verbose)
    y_clean = y.loc[keep_mask].reset_index(drop=True)

    if verbose and method == "drop":
        print(f"{label}: kept {len(y_clean):,} of {len(y):,} rows after outlier filtering")

    return X_clean, y_clean, summary


# -----------------------------------------------------------------------------
# Dummy coding for modeling, drops columns used to build dummies and unused booleans
# -----------------------------------------------------------------------------
def encode_model_features(df, verbose=True):
    work_df = df.copy()
    work_df["log_tip"] = np.log1p(work_df["tip_amount"])

    hour_bins = [-1, 3, 6, 15, 20, 24]
    hour_labels = [
        "Late Night",
        "Airport Rush",
        "Daytime",
        "Evening Commute",
        "Nightlife"
    ]
    work_df["hour_bin"] = pd.cut(work_df["pickup_hour"], bins=hour_bins, labels=hour_labels, right=False)
    work_df["hour_bin"] = pd.Categorical(work_df["hour_bin"], categories=hour_labels, ordered=True)
    hour_dummies = pd.get_dummies(work_df["hour_bin"], prefix="hour", drop_first=True, dtype=np.int8)

    month_map = {i: calendar.month_name[i] for i in range(1, 13)}
    month_order = [calendar.month_name[i] for i in range(1, 13)]
    work_df["month_cat"] = work_df["month_num"].map(month_map)
    work_df["month_cat"] = pd.Categorical(work_df["month_cat"], categories=month_order, ordered=True)
    month_dummies = pd.get_dummies(work_df["month_cat"], prefix="month", drop_first=True, dtype=np.int8)

    borough_order = [
        "Bronx",
        "Brooklyn",
        "Manhattan",
        "Queens",
        "Staten Island",
        "EWR",
        "Unknown"
    ]
    observed = set(work_df["pickup_borough"].astype(str).dropna().unique())
    extras = sorted(observed - set(borough_order))
    borough_order_full = borough_order + extras
    work_df["pickup_borough_cat"] = pd.Categorical(
        work_df["pickup_borough"].astype(str),
        categories=borough_order_full,
        ordered=False
    )
    borough_dummies = pd.get_dummies(work_df["pickup_borough_cat"], prefix="borough", drop_first=True, dtype=np.int8)

    work_df = pd.concat([work_df, hour_dummies, month_dummies, borough_dummies], axis=1)

    drop_cols = [
        "pickup_hour",
        "month_num",
        "pickup_borough",
        "hour_bin",
        "month_cat",
        "pickup_borough_cat",
        "dow",
        "rush_hour",
        "night_trip"
    ]
    work_df = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])

    if "weekend" in work_df.columns:
        work_df["weekend"] = work_df["weekend"].astype(np.int8)

    retained_binary_features = [c for c in ["weekend", "is_airport_trip"] if c in work_df.columns]
    new_features = retained_binary_features + list(hour_dummies.columns) + list(month_dummies.columns) + list(
        borough_dummies.columns
    )
    if verbose:
        print(f"Encoded {len(new_features)} engineered features.")
    return work_df, new_features


# -----------------------------------------------------------------------------
# Build leakage-safe model matrix for selected target
# -----------------------------------------------------------------------------
def build_model_matrix(df, target="tip_amount", verbose=True):
    _validate_target(target)

    y = pd.to_numeric(df[target], errors="coerce")
    keep_rows = y.notna()
    y = y.loc[keep_rows].reset_index(drop=True)

    # Explicitly remove target and target-adjacent columns from predictors
    candidate_cols = [c for c in df.columns if c not in LEAKAGE_DROP_COLS]
    X = df.loc[keep_rows, candidate_cols].copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)

    bad_types = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad_types:
        raise ValueError(f"Non-numeric columns remained in model matrix: {bad_types}")

    if verbose:
        print(f"Target: {target}")
        print(f"X shape: {X.shape[0]:,} rows x {X.shape[1]:,} columns")
        print(f"Leakage-safe dropped columns: {sorted(LEAKAGE_DROP_COLS)}")

    return X.reset_index(drop=True), y


# -----------------------------------------------------------------------------
# Feature grouping helper for diagnostics and preprocessing decisions
# -----------------------------------------------------------------------------
def split_feature_groups(X, dummy_prefixes=DUMMY_PREFIXES):
    dummy_cols = [c for c in X.columns if any(c.startswith(p) for p in dummy_prefixes)]
    binary_cols = []
    low_card_cols = []
    continuous_cols = []

    for c in X.columns:
        if c in dummy_cols:
            continue

        nunique = int(X[c].dropna().nunique())
        if nunique <= 2:
            binary_cols.append(c)
        elif nunique <= 10:
            low_card_cols.append(c)
        else:
            continuous_cols.append(c)

    return {
        "dummy_cols": dummy_cols,
        "binary_cols": binary_cols,
        "low_card_cols": low_card_cols,
        "continuous_cols": continuous_cols
    }


# -----------------------------------------------------------------------------
# Profile target shape and basic transform signal
# -----------------------------------------------------------------------------
def check_target_profile(y, target, show_plots=False):
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    if y_num.empty:
        raise ValueError(f"Target '{target}' has no numeric values.")

    desc = y_num.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    skew = float(y_num.skew())
    kurt = float(y_num.kurtosis())
    zero_pct = float((y_num == 0).mean() * 100)
    negative_pct = float((y_num < 0).mean() * 100)
    suggest_log1p = abs(skew) >= 1.0 and target != "log_tip" and negative_pct == 0.0

    print(f"\nTarget profile: {target}")
    print(desc.to_string())
    print(f"Skew={skew:.3f} | Kurtosis={kurt:.3f} | Zero rate={zero_pct:.2f}% | Negative rate={negative_pct:.2f}%")
    if suggest_log1p:
        print("Transform guidance: strong skew and non-negative target; log1p target is recommended.")
    else:
        print("Transform guidance: no target transform is strictly required by skew alone.")

    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        sns.histplot(y_num, kde=True, bins=80, color="steelblue", ax=axes[0])
        axes[0].set_title(f"{target} distribution")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        sns.boxplot(x=y_num, color="lightcoral", ax=axes[1])
        axes[1].set_title(f"{target} boxplot")
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.4)

        stats.probplot(y_num, dist="norm", plot=axes[2])
        if len(axes[2].get_lines()) > 1:
            axes[2].get_lines()[1].set_color("red")
        axes[2].set_title(f"{target} Q-Q")
        axes[2].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    return {
        "target": target,
        "n": int(len(y_num)),
        "skew": skew,
        "kurtosis": kurt,
        "zero_pct": zero_pct,
        "negative_pct": negative_pct,
        "suggest_log1p_target": bool(suggest_log1p)
    }


# -----------------------------------------------------------------------------
# Correlation summary between predictors and target
# -----------------------------------------------------------------------------
def feature_target_correlations(X, y):
    y_num = pd.to_numeric(y, errors="coerce")
    rows = []

    for c in X.columns:
        x_num = pd.to_numeric(X[c], errors="coerce")
        valid = x_num.notna() & y_num.notna()
        if valid.sum() < 3:
            continue

        x_valid = x_num.loc[valid]
        y_valid = y_num.loc[valid]
        if x_valid.nunique() <= 1 or y_valid.nunique() <= 1:
            continue

        pearson = x_valid.corr(y_valid, method="pearson")
        spearman = x_valid.corr(y_valid, method="spearman")
        rows.append({
            "feature": c,
            "pearson_corr": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman_corr": float(spearman) if pd.notna(spearman) else np.nan
        })

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df

    corr_df["abs_pearson"] = corr_df["pearson_corr"].abs()
    corr_df["abs_spearman"] = corr_df["spearman_corr"].abs()
    return corr_df.sort_values(["abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Skew profiling with transform recommendations on continuous features
# -----------------------------------------------------------------------------
def profile_feature_skew(X, cols, skew_threshold=1.0):
    rows = []

    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        if s.empty:
            continue

        q01, q25, q75, q99 = s.quantile([0.01, 0.25, 0.75, 0.99])
        iqr = q75 - q25
        tail_ratio = np.nan if iqr == 0 else float((q99 - q01) / iqr)
        skew = float(s.skew())

        suggested_transform = "none"
        if abs(skew) >= skew_threshold:
            suggested_transform = "log1p" if s.min() >= 0 else "yeo-johnson"

        rows.append({
            "feature": c,
            "skew": skew,
            "abs_skew": abs(skew),
            "q01": float(q01),
            "q99": float(q99),
            "min": float(s.min()),
            "max": float(s.max()),
            "tail_ratio_99_01_to_iqr": tail_ratio,
            "suggested_transform": suggested_transform
        })

    skew_df = pd.DataFrame(rows)
    if skew_df.empty:
        return skew_df

    return skew_df.sort_values("abs_skew", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Detect high pairwise predictor correlation candidates
# -----------------------------------------------------------------------------
def high_feature_correlations(X, cols, threshold=0.9):
    usable_cols = [c for c in cols if X[c].dropna().nunique() > 1]
    if len(usable_cols) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "abs_corr"])

    corr_matrix = X[usable_cols].corr(method="pearson").abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = upper_triangle.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    pairs = pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return pairs[pairs["abs_corr"] >= threshold].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Identify zero and near-zero variance predictors
# -----------------------------------------------------------------------------
def low_variance_report(X, zero_var_threshold=1e-12, near_zero_var_threshold=1e-4):
    variances = X.var(numeric_only=True)
    zero_var_features = variances[variances <= zero_var_threshold].index.tolist()
    near_zero_var_features = variances[
        (variances > zero_var_threshold) & (variances <= near_zero_var_threshold)
    ].index.tolist()

    return {
        "zero_var_features": zero_var_features,
        "near_zero_var_features": near_zero_var_features
    }


# -----------------------------------------------------------------------------
# Scaling recommendations based on spread and skew
# -----------------------------------------------------------------------------
def scaling_recommendation(X, continuous_cols, skew_df, std_ratio_cutoff=20.0, range_ratio_cutoff=100.0):
    model_guidance = {
        "linear_models": "scale_continuous_features",
        "distance_based_models": "scale_all_numeric_features",
        "tree_models": "scaling_not_required"
    }

    if not continuous_cols:
        return {
            "recommend_scaling": False,
            "recommended_scaler": "none",
            "std_ratio": np.nan,
            "range_ratio": np.nan,
            "skew_flag_rate_pct": np.nan,
            "reason": "No continuous features detected.",
            "model_guidance": model_guidance
        }

    cont = X[continuous_cols]
    std_series = cont.std()
    range_series = cont.max() - cont.min()

    std_nonzero = std_series[std_series > 0]
    range_nonzero = range_series[range_series > 0]
    std_ratio = float(std_nonzero.max() / std_nonzero.min()) if len(std_nonzero) > 0 else np.nan
    range_ratio = float(range_nonzero.max() / range_nonzero.min()) if len(range_nonzero) > 0 else np.nan

    skew_flag_rate = 0.0
    if not skew_df.empty:
        skew_flag_rate = float((skew_df["suggested_transform"] != "none").mean() * 100)

    recommend_scaling = (pd.notna(std_ratio) and std_ratio >= std_ratio_cutoff) or (
        pd.notna(range_ratio) and range_ratio >= range_ratio_cutoff
    )

    if not recommend_scaling:
        recommended_scaler = "optional_for_linear_models"
        reason = "Scale differences are moderate; scaling is optional unless model class requires it."
    else:
        recommended_scaler = "StandardScaler"
        if skew_flag_rate >= 30:
            reason = "Large scale differences with heavy skew; standard scaling after log transform is recommended."
        else:
            reason = "Large scale differences with moderate skew; standard scaling is appropriate."

    return {
        "recommend_scaling": bool(recommend_scaling),
        "recommended_scaler": recommended_scaler,
        "std_ratio": std_ratio,
        "range_ratio": range_ratio,
        "skew_flag_rate_pct": skew_flag_rate,
        "reason": reason,
        "model_guidance": model_guidance
    }


# -----------------------------------------------------------------------------
# Full pre-modeling diagnostics bundle
# -----------------------------------------------------------------------------
def run_model_prep_checks(X, y, target, top_n=15, corr_threshold=0.9, skew_threshold=1.0, show_plots=False):
    groups = split_feature_groups(X)
    target_stats = check_target_profile(y, target=target, show_plots=show_plots)
    corr_df = feature_target_correlations(X, y)
    skew_df = profile_feature_skew(X, groups["continuous_cols"], skew_threshold=skew_threshold)
    high_corr_pairs = high_feature_correlations(
        X,
        cols=groups["continuous_cols"] + groups["low_card_cols"],
        threshold=corr_threshold
    )
    low_var = low_variance_report(X)
    scale_info = scaling_recommendation(X, groups["continuous_cols"], skew_df)

    print("\nFeature groups")
    print(f"Continuous: {len(groups['continuous_cols'])} | Low-card: {len(groups['low_card_cols'])} | "
          f"Binary: {len(groups['binary_cols'])} | Dummies: {len(groups['dummy_cols'])}")

    if not corr_df.empty:
        print(f"\nTop {top_n} feature-target relationships (|Spearman|):")
        print(
            corr_df[["feature", "spearman_corr", "pearson_corr"]]
            .head(top_n)
            .round(4)
            .to_string(index=False)
        )
    else:
        print("\nFeature-target relationships: no valid numeric pairs.")

    if not skew_df.empty:
        print(f"\nTop {top_n} skewed continuous features:")
        print(
            skew_df[["feature", "skew", "suggested_transform", "q01", "q99"]]
            .head(top_n)
            .round(4)
            .to_string(index=False)
        )
    else:
        print("\nSkew profile: no continuous features detected.")

    if len(high_corr_pairs) > 0:
        print(f"\nHigh predictor correlation pairs (>= {corr_threshold}):")
        print(high_corr_pairs.head(top_n).round(4).to_string(index=False))
    else:
        print(f"\nHigh predictor correlation pairs (>= {corr_threshold}): none")

    print("\nLow variance features")
    print(f"Zero variance: {low_var['zero_var_features']}")
    print(f"Near-zero variance: {low_var['near_zero_var_features']}")

    print("\nScaling guidance")
    print(f"Recommend scaling: {scale_info['recommend_scaling']}")
    print(f"Recommended scaler: {scale_info['recommended_scaler']}")
    print(f"Std ratio: {scale_info['std_ratio']:.3f} | Range ratio: {scale_info['range_ratio']:.3f}")
    print(f"Reason: {scale_info['reason']}")
    print("Model-family guidance:")
    print(f"  linear_models: {scale_info['model_guidance']['linear_models']}")
    print(f"  distance_based_models: {scale_info['model_guidance']['distance_based_models']}")
    print(f"  tree_models: {scale_info['model_guidance']['tree_models']}")

    if show_plots and not corr_df.empty:
        top_corr_cols = corr_df["feature"].head(min(12, len(corr_df))).tolist()
        if len(top_corr_cols) >= 2:
            corr_plot = X[top_corr_cols].corr(method="pearson")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_plot, cmap="coolwarm", center=0, square=True)
            plt.title("Top feature correlation heatmap")
            plt.tight_layout()
            plt.show()

    return {
        "target_stats": target_stats,
        "feature_groups": groups,
        "feature_target_corr": corr_df,
        "skew_table": skew_df,
        "high_corr_pairs": high_corr_pairs,
        "low_variance": low_var,
        "scaling": scale_info
    }


# -----------------------------------------------------------------------------
# Simple preprocessing utilities for modeling
# -----------------------------------------------------------------------------
def choose_log1p_columns(X_train, prep_report=None):
    if prep_report is not None and "skew_table" in prep_report and not prep_report["skew_table"].empty:
        skew_table = prep_report["skew_table"]
        log_cols = skew_table.loc[skew_table["suggested_transform"] == "log1p", "feature"].tolist()
        return [c for c in log_cols if c in X_train.columns]

    groups = split_feature_groups(X_train)
    log_cols = []
    for c in groups["continuous_cols"]:
        s = pd.to_numeric(X_train[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        if s.min() >= 0 and abs(float(s.skew())) >= 1.0:
            log_cols.append(c)
    return log_cols


def choose_near_zero_columns(X_train, threshold=1e-4):
    variances = X_train.var(numeric_only=True)
    return variances[(variances > 0) & (variances <= threshold)].index.tolist()


def choose_high_corr_drop_columns(X_train, prep_report=None, corr_threshold=0.9):
    if prep_report is not None and "high_corr_pairs" in prep_report:
        high_corr_pairs = prep_report["high_corr_pairs"].copy()
    else:
        high_corr_pairs = high_feature_correlations(X_train, cols=X_train.columns.tolist(), threshold=corr_threshold)

    if len(high_corr_pairs) == 0:
        return []

    existing_cols = set(X_train.columns)
    high_corr_pairs = high_corr_pairs[
        high_corr_pairs["feature_1"].isin(existing_cols) & high_corr_pairs["feature_2"].isin(existing_cols)
    ].copy()
    if len(high_corr_pairs) == 0:
        return []

    high_corr_pairs = high_corr_pairs[high_corr_pairs["abs_corr"] >= corr_threshold].copy()
    if len(high_corr_pairs) == 0:
        return []

    target_strength = {}
    if prep_report is not None and "feature_target_corr" in prep_report and len(prep_report["feature_target_corr"]) > 0:
        corr_table = prep_report["feature_target_corr"]
        strength_col = "abs_spearman" if "abs_spearman" in corr_table.columns else "abs_pearson"
        target_strength = corr_table.set_index("feature")[strength_col].to_dict()

    variance_lookup = X_train.var(numeric_only=True).to_dict()
    drop_cols = []

    for _, row in high_corr_pairs.iterrows():
        f1 = row["feature_1"]
        f2 = row["feature_2"]

        if f1 in drop_cols or f2 in drop_cols:
            continue

        s1 = target_strength.get(f1, np.nan)
        s2 = target_strength.get(f2, np.nan)

        # Prefer keeping the feature with stronger target relationship, fallback to higher variance
        if pd.notna(s1) and pd.notna(s2):
            drop_feature = f1 if s1 < s2 else f2
        else:
            v1 = variance_lookup.get(f1, np.nan)
            v2 = variance_lookup.get(f2, np.nan)
            if pd.notna(v1) and pd.notna(v2):
                drop_feature = f1 if v1 < v2 else f2
            else:
                drop_feature = sorted([f1, f2])[1]

        drop_cols.append(drop_feature)

    return sorted(set(drop_cols))


def preprocess_for_model(
        X_train,
        X_valid,
        prep_report=None,
        apply_log=False,
        apply_scale=False,
        drop_near_zero=False,
        near_zero_var_threshold=1e-4,
        drop_high_corr=False,
        corr_threshold=0.9,
        verbose=True):
    X_train_proc = X_train.copy()
    X_valid_proc = X_valid.copy()

    groups = prep_report["feature_groups"] if prep_report is not None else split_feature_groups(X_train_proc)
    continuous_cols = [c for c in groups["continuous_cols"] if c in X_train_proc.columns]
    log_cols = choose_log1p_columns(X_train_proc, prep_report=prep_report) if apply_log else []
    scale_cols = continuous_cols if apply_scale else []

    # Order is intentional: transform -> scale -> prune unstable predictors
    if len(log_cols) > 0:
        for c in log_cols:
            X_train_proc[c] = np.log1p(np.clip(X_train_proc[c], a_min=0, a_max=None))
            X_valid_proc[c] = np.log1p(np.clip(X_valid_proc[c], a_min=0, a_max=None))

    if len(scale_cols) > 0:
        scaler = StandardScaler()
        for c in scale_cols:
            X_train_proc[c] = X_train_proc[c].astype(float)
            X_valid_proc[c] = X_valid_proc[c].astype(float)
        # Fit scaler on train only, then transform valid with frozen parameters
        X_train_proc.loc[:, scale_cols] = scaler.fit_transform(X_train_proc[scale_cols])
        X_valid_proc.loc[:, scale_cols] = scaler.transform(X_valid_proc[scale_cols])

    constant_cols = [c for c in X_train_proc.columns if X_train_proc[c].nunique(dropna=False) <= 1]
    if len(constant_cols) > 0:
        X_train_proc = X_train_proc.drop(columns=constant_cols)
        X_valid_proc = X_valid_proc.drop(columns=constant_cols)

    near_zero_cols = []
    if drop_near_zero:
        near_zero_cols = choose_near_zero_columns(X_train_proc, threshold=near_zero_var_threshold)
        if len(near_zero_cols) > 0:
            X_train_proc = X_train_proc.drop(columns=near_zero_cols)
            X_valid_proc = X_valid_proc.drop(columns=near_zero_cols)

    high_corr_drop_cols = []
    if drop_high_corr:
        high_corr_drop_cols = choose_high_corr_drop_columns(
            X_train_proc,
            prep_report=prep_report,
            corr_threshold=corr_threshold
        )
        if len(high_corr_drop_cols) > 0:
            X_train_proc = X_train_proc.drop(columns=high_corr_drop_cols)
            X_valid_proc = X_valid_proc.drop(columns=high_corr_drop_cols)

    prep_info = {
        "apply_log": bool(apply_log),
        "apply_scale": bool(apply_scale),
        "drop_near_zero": bool(drop_near_zero),
        "drop_high_corr": bool(drop_high_corr),
        "log_cols": log_cols,
        "scale_cols": scale_cols,
        "constant_cols_dropped": constant_cols,
        "near_zero_cols_dropped": near_zero_cols,
        "high_corr_cols_dropped": high_corr_drop_cols,
        "n_features_out": int(X_train_proc.shape[1])
    }

    if verbose:
        print("\nPreprocessing summary")
        print(
            f"log1p cols: {len(log_cols)} | scaled cols: {len(scale_cols)} | "
            f"dropped constants: {len(constant_cols)} | dropped near-zero: {len(near_zero_cols)} | "
            f"dropped high-corr: {len(high_corr_drop_cols)}"
        )

    return X_train_proc, X_valid_proc, prep_info


def print_model_prep_qa(model_name, prep_info, max_show=8):
    def _short(cols):
        if len(cols) == 0:
            return "[]"
        shown = cols[:max_show]
        suffix = "" if len(cols) <= max_show else f" ... (+{len(cols) - max_show} more)"
        return "[" + ", ".join(shown) + "]" + suffix

    constant_cols = prep_info.get("constant_cols_dropped", [])
    near_zero_cols = prep_info.get("near_zero_cols_dropped", [])
    high_corr_cols = prep_info.get("high_corr_cols_dropped", [])
    log_cols = prep_info.get("log_cols", [])
    scale_cols = prep_info.get("scale_cols", [])

    print(f"\nPrep QA | {model_name}")
    print(
        f"features_out={prep_info.get('n_features_out', np.nan)} | "
        f"log1p_cols={len(log_cols)} | scaled_cols={len(scale_cols)}"
    )
    print(f"dropped constants ({len(constant_cols)}): {_short(constant_cols)}")
    print(f"dropped near-zero ({len(near_zero_cols)}): {_short(near_zero_cols)}")
    print(f"dropped high-corr ({len(high_corr_cols)}): {_short(high_corr_cols)}")


# -----------------------------------------------------------------------------
# Modeling metrics
# -----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred))
    }


def classification_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = np.nan
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["avg_precision"] = np.nan
    return metrics


def fit_naive_regression_baseline(y_train, y_valid, positive_only=True, verbose=True):
    y_train_num = pd.to_numeric(y_train, errors="coerce")
    y_valid_num = pd.to_numeric(y_valid, errors="coerce")
    if positive_only:
        y_train_num = y_train_num[y_train_num > 0]
        y_valid_num = y_valid_num[y_valid_num > 0]
    y_train_num = y_train_num.dropna().reset_index(drop=True)
    y_valid_num = y_valid_num.dropna().reset_index(drop=True)

    if len(y_train_num) == 0 or len(y_valid_num) == 0:
        raise ValueError("Naive regression baseline requires non-empty train and valid targets")

    baseline_pred = float(y_train_num.mean())
    y_pred = np.full(len(y_valid_num), baseline_pred, dtype=float)
    metrics = regression_metrics(y_valid_num, y_pred)
    pred_df = pd.DataFrame({"y_true": y_valid_num, "y_pred": y_pred})
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]

    summary = {
        "model_name": "naive_mean_tip_regression",
        "fit_seconds": 0.0,
        "train_rows": int(len(y_train_num)),
        "valid_rows": int(len(y_valid_num)),
        "baseline_pred_value": baseline_pred,
        "metrics": metrics
    }

    if verbose:
        print("\nRegression baseline: Naive mean")
        print(f"Baseline prediction={baseline_pred:.4f} | RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")

    return {
        "model": None,
        "summary": summary,
        "valid_predictions": pred_df
    }


def fit_naive_classification_baseline(y_train, y_valid, verbose=True):
    y_train_bin = (pd.to_numeric(y_train, errors="coerce") > 0).astype(int).reset_index(drop=True)
    y_valid_bin = (pd.to_numeric(y_valid, errors="coerce") > 0).astype(int).reset_index(drop=True)
    if y_train_bin.nunique() == 0 or y_valid_bin.nunique() == 0:
        raise ValueError("Naive classification baseline requires non-empty train and valid targets")

    train_tip_rate = float(y_train_bin.mean())
    majority_class = int(train_tip_rate >= 0.5)
    y_pred = np.full(len(y_valid_bin), majority_class, dtype=int)
    y_prob = np.full(len(y_valid_bin), train_tip_rate, dtype=float)
    metrics = classification_metrics(y_valid_bin, y_pred, y_prob)
    pred_df = pd.DataFrame({"y_true": y_valid_bin, "y_pred": y_pred, "y_prob": y_prob})

    summary = {
        "model_name": "naive_majority_tip_classifier",
        "fit_seconds": 0.0,
        "train_rows": int(len(y_train_bin)),
        "valid_rows": int(len(y_valid_bin)),
        "majority_class": majority_class,
        "train_tip_rate": train_tip_rate,
        "metrics": metrics
    }

    if verbose:
        print("\nClassification baseline: Naive majority class")
        print(
            f"Majority class={majority_class} | Train tip rate={train_tip_rate:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | AP={metrics['avg_precision']:.4f} | F1={metrics['f1']:.4f}"
        )

    return {
        "model": None,
        "summary": summary,
        "valid_predictions": pred_df
    }


def tune_lightgbm_regressor(
        X_train,
        y_train,
        seed=42,
        n_trials=40,
        timeout_seconds=None,
        inner_valid_frac=0.2,
        early_stop_rounds=50,
        verbose=True):
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive (got {n_trials})")
    if not (0 < inner_valid_frac < 1):
        raise ValueError(f"inner_valid_frac must be in (0,1), got {inner_valid_frac}")

    # Nested holdout inside train fold for parameter search only
    X_tune_train, X_tune_valid, y_tune_train, y_tune_valid = train_test_split(
        X_train,
        y_train,
        test_size=inner_valid_frac,
        random_state=seed
    )
    X_tune_train = X_tune_train.reset_index(drop=True)
    X_tune_valid = X_tune_valid.reset_index(drop=True)
    y_tune_train = y_tune_train.reset_index(drop=True)
    y_tune_valid = y_tune_valid.reset_index(drop=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Minimize RMSE on the inner validation split
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        }
        model = LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
            **params
        )
        model.fit(
            X_tune_train,
            y_tune_train,
            eval_set=[(X_tune_valid, y_tune_valid)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=early_stop_rounds, verbose=False)]
        )
        y_pred = model.predict(X_tune_valid)
        rmse = float(np.sqrt(mean_squared_error(y_tune_valid, y_pred))) # type: ignore
        trial.set_user_attr("best_iteration", int(model.best_iteration_) if model.best_iteration_ else np.nan)
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    tune_seconds = float(time.perf_counter() - start)

    best_params = study.best_params.copy()
    best_rmse = float(study.best_value)
    best_iteration = study.best_trial.user_attrs.get("best_iteration", np.nan)

    if verbose:
        print("\nOptuna tuning summary")
        print(
            f"Trials run: {len(study.trials)} | Best inner RMSE: {best_rmse:.4f} | "
            f"Tuning time: {tune_seconds:.2f}s | Best iteration: {best_iteration}"
        )
        print("Best params:")
        print(best_params)

    return {
        "best_params": best_params,
        "best_rmse": best_rmse,
        "best_iteration": best_iteration,
        "n_trials_run": int(len(study.trials)),
        "tune_seconds": tune_seconds
    }


def tune_random_forest_classifier(
        X_train,
        y_train_bin,
        seed=42,
        n_trials=30,
        timeout_seconds=None,
        inner_valid_frac=0.2,
        max_tune_rows=150_000,
        verbose=True):
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive (got {n_trials})")
    if not (0 < inner_valid_frac < 1):
        raise ValueError(f"inner_valid_frac must be in (0,1), got {inner_valid_frac}")
    if max_tune_rows is not None and max_tune_rows <= 1000:
        raise ValueError(f"max_tune_rows must be > 1000 when provided (got {max_tune_rows})")
    if y_train_bin.nunique() < 2:
        raise ValueError("Random forest tuning requires both classes in y_train_bin")

    X_work = X_train.reset_index(drop=True)
    y_work = y_train_bin.reset_index(drop=True)

    # Cap tuning rows to keep Optuna runtime manageable on large yearly samples
    if max_tune_rows is not None and len(X_work) > max_tune_rows:
        X_work, _, y_work, _ = train_test_split(
            X_work,
            y_work,
            train_size=max_tune_rows,
            random_state=seed,
            stratify=y_work
        )
        X_work = X_work.reset_index(drop=True)
        y_work = y_work.reset_index(drop=True)

    X_tune_train, X_tune_valid, y_tune_train, y_tune_valid = train_test_split(
        X_work,
        y_work,
        test_size=inner_valid_frac,
        random_state=seed,
        stratify=y_work
    )
    X_tune_train = X_tune_train.reset_index(drop=True)
    X_tune_valid = X_tune_valid.reset_index(drop=True)
    y_tune_train = y_tune_train.reset_index(drop=True)
    y_tune_valid = y_tune_valid.reset_index(drop=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Maximize ROC-AUC on inner validation for class-separation quality
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        }
        model = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params
        )
        model.fit(X_tune_train, y_tune_train)
        y_prob = model.predict_proba(X_tune_valid)[:, 1]
        try:
            auc = float(roc_auc_score(y_tune_valid, y_prob))
        except ValueError:
            auc = 0.5
        return auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )

    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    tune_seconds = float(time.perf_counter() - start)

    best_params = study.best_params.copy()
    best_auc = float(study.best_value)

    if verbose:
        print("\nOptuna RF tuning summary")
        print(
            f"Trials run: {len(study.trials)} | Best inner AUC: {best_auc:.4f} | "
            f"Tuning time: {tune_seconds:.2f}s | Tune rows: {len(X_work):,}"
        )
        print("Best params:")
        print(best_params)

    return {
        "best_params": best_params,
        "best_auc": best_auc,
        "n_trials_run": int(len(study.trials)),
        "tune_seconds": tune_seconds,
        "tune_rows": int(len(X_work))
    }


# -----------------------------------------------------------------------------
# Requested simple and optimized models
# -----------------------------------------------------------------------------
def fit_simple_linear_regression(X_train, y_train, X_valid, y_valid, prep_report=None, verbose=True):
    # Restrict to tipping trips so coefficients are interpretable on positive tip dollars
    pos_train = y_train > 0
    pos_valid = y_valid > 0

    X_train_use, X_valid_use, prep_info = preprocess_for_model(
        X_train=X_train.loc[pos_train].reset_index(drop=True),
        X_valid=X_valid.loc[pos_valid].reset_index(drop=True),
        prep_report=prep_report,
        apply_log=False,
        apply_scale=False,
        drop_near_zero=True,
        near_zero_var_threshold=1e-4,
        drop_high_corr=True,
        corr_threshold=0.9,
        verbose=False
    )
    y_train_use = y_train.loc[pos_train].reset_index(drop=True)
    y_valid_use = y_valid.loc[pos_valid].reset_index(drop=True)

    if verbose:
        print_model_prep_qa("simple_linear_regression", prep_info, max_show=10)

    start = time.perf_counter()
    model = LinearRegression()
    model.fit(X_train_use, y_train_use)
    y_pred = model.predict(X_valid_use)
    fit_seconds = float(time.perf_counter() - start)

    metrics = regression_metrics(y_valid_use, y_pred)
    coef_table = pd.DataFrame({"feature": X_train_use.columns, "coef": model.coef_})
    coef_table["abs_coef"] = coef_table["coef"].abs()
    coef_table = coef_table.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame({"y_true": y_valid_use.reset_index(drop=True), "y_pred": y_pred})
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]

    summary = {
        "model_name": "simple_linear_regression",
        "fit_seconds": fit_seconds,
        "train_rows": int(len(X_train_use)),
        "valid_rows": int(len(X_valid_use)),
        "intercept": float(model.intercept_),
        "metrics": metrics,
        "prep_info": prep_info
    }

    if verbose:
        print("\nModel 1: Simple linear regression")
        print(f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")

    return {
        "model": model,
        "summary": summary,
        "coef_table": coef_table,
        "valid_predictions": pred_df
    }


def fit_optimized_regression_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=None,
        seed=42,
        tune=True,
        optuna_trials=40,
        optuna_timeout=None,
        verbose=True):
    pos_train = y_train > 0
    pos_valid = y_valid > 0

    X_train_use, X_valid_use, prep_info = preprocess_for_model(
        X_train=X_train.loc[pos_train].reset_index(drop=True),
        X_valid=X_valid.loc[pos_valid].reset_index(drop=True),
        prep_report=prep_report,
        apply_log=True,
        apply_scale=True,
        drop_near_zero=False,
        drop_high_corr=False,
        verbose=False
    )
    y_train_use = y_train.loc[pos_train].reset_index(drop=True)
    y_valid_use = y_valid.loc[pos_valid].reset_index(drop=True)

    if verbose:
        print_model_prep_qa("optimized_lightgbm_regression", prep_info, max_show=10)

    X_train_use = X_train_use.rename(columns=lambda c: c.replace(" ", "_"))
    X_valid_use = X_valid_use.rename(columns=lambda c: c.replace(" ", "_"))

    tune_report = None
    if tune:
        tune_report = tune_lightgbm_regressor(
            X_train=X_train_use,
            y_train=y_train_use,
            seed=seed,
            n_trials=optuna_trials,
            timeout_seconds=optuna_timeout,
            inner_valid_frac=0.2,
            early_stop_rounds=50,
            verbose=verbose
        )
        model_params = tune_report["best_params"].copy()
    else:
        model_params = {k: v for k, v in DEFAULT_LGBM_PARAMS.items() if k != "objective"}

    model = LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        **model_params
    )
    model_name = "optimized_lightgbm_regression"

    start = time.perf_counter()
    model.fit(
        X_train_use,
        y_train_use,
        eval_set=[(X_valid_use, y_valid_use)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=75, verbose=False)]
    )
    y_pred = model.predict(X_valid_use)
    fit_seconds = float(time.perf_counter() - start)
    metrics = regression_metrics(y_valid_use, y_pred)
    importance_table = pd.DataFrame({"feature": X_train_use.columns, "importance": model.feature_importances_})
    importance_table = importance_table.sort_values("importance", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame({"y_true": y_valid_use.reset_index(drop=True), "y_pred": y_pred})
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]

    summary = {
        "model_name": model_name,
        "fit_seconds": fit_seconds,
        "train_rows": int(len(X_train_use)),
        "valid_rows": int(len(X_valid_use)),
        "metrics": metrics,
        "tuned_with_optuna": bool(tune),
        "optuna_tuning_seconds": float(tune_report["tune_seconds"]) if tune_report is not None else 0.0,
        "optuna_trials_run": int(tune_report["n_trials_run"]) if tune_report is not None else 0,
        "optuna_best_inner_rmse": float(tune_report["best_rmse"]) if tune_report is not None else np.nan,
        "model_params": model_params,
        "prep_info": prep_info
    }

    if verbose:
        print("\nModel 2: Optimized regression model")
        print("Estimator: LightGBM regressor")
        if tune:
            print(
                f"Optuna tuned: trials={summary['optuna_trials_run']} | "
                f"best inner RMSE={summary['optuna_best_inner_rmse']:.4f} | "
                f"tune_seconds={summary['optuna_tuning_seconds']:.2f}"
            )
        print(f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")

    return {
        "model": model,
        "summary": summary,
        "feature_importance": importance_table,
        "valid_predictions": pred_df
    }


def fit_simple_logistic_regression(X_train, y_train, X_valid, y_valid, prep_report=None, verbose=True):
    y_train_bin = (y_train > 0).astype(int).reset_index(drop=True)
    y_valid_bin = (y_valid > 0).astype(int).reset_index(drop=True)

    X_train_use, X_valid_use, prep_info = preprocess_for_model(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        prep_report=prep_report,
        apply_log=False,
        apply_scale=False,
        drop_near_zero=True,
        near_zero_var_threshold=1e-4,
        drop_high_corr=True,
        corr_threshold=0.9,
        verbose=False
    )

    if verbose:
        print_model_prep_qa("simple_logistic_regression", prep_info, max_show=10)

    start = time.perf_counter()
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X_train_use, y_train_bin)
    y_pred = model.predict(X_valid_use)
    y_prob = model.predict_proba(X_valid_use)[:, 1]
    fit_seconds = float(time.perf_counter() - start)
    metrics = classification_metrics(y_valid_bin, y_pred, y_prob)

    coef_table = pd.DataFrame({"feature": X_train_use.columns, "log_odds_coef": model.coef_[0]})
    coef_table["odds_ratio"] = np.exp(coef_table["log_odds_coef"])
    coef_table["abs_log_odds_coef"] = coef_table["log_odds_coef"].abs()
    coef_table = coef_table.sort_values("abs_log_odds_coef", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame({"y_true": y_valid_bin, "y_pred": y_pred, "y_prob": y_prob})

    summary = {
        "model_name": "simple_logistic_regression",
        "fit_seconds": fit_seconds,
        "train_rows": int(len(X_train_use)),
        "valid_rows": int(len(X_valid_use)),
        "intercept_log_odds": float(model.intercept_[0]),
        "metrics": metrics,
        "prep_info": prep_info
    }

    if verbose:
        print("\nModel 3: Simple logistic regression")
        print(f"AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}")

    return {
        "model": model,
        "summary": summary,
        "coef_table": coef_table,
        "valid_predictions": pred_df
    }


def fit_optimized_random_forest_classifier(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=None,
        seed=42,
        tune=True,
        optuna_trials=30,
        optuna_timeout=None,
        rf_max_tune_rows=150_000,
        verbose=True):
    y_train_bin = (y_train > 0).astype(int).reset_index(drop=True)
    y_valid_bin = (y_valid > 0).astype(int).reset_index(drop=True)

    X_train_use, X_valid_use, prep_info = preprocess_for_model(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        prep_report=prep_report,
        # Log transform handles heavy right tails, scaling is not needed for tree splits
        apply_log=True,
        apply_scale=False,
        drop_near_zero=False,
        drop_high_corr=False,
        verbose=False
    )

    if verbose:
        print_model_prep_qa("optimized_random_forest_classifier", prep_info, max_show=10)

    tune_report = None
    if tune:
        tune_report = tune_random_forest_classifier(
            X_train=X_train_use,
            y_train_bin=y_train_bin,
            seed=seed,
            n_trials=optuna_trials,
            timeout_seconds=optuna_timeout,
            inner_valid_frac=0.2,
            max_tune_rows=rf_max_tune_rows,
            verbose=verbose
        )
        model_params = tune_report["best_params"].copy()
    else:
        model_params = DEFAULT_RF_PARAMS.copy()

    start = time.perf_counter()
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **model_params)
    model.fit(X_train_use, y_train_bin)
    y_pred = model.predict(X_valid_use)
    y_prob = model.predict_proba(X_valid_use)[:, 1]
    fit_seconds = float(time.perf_counter() - start)
    metrics = classification_metrics(y_valid_bin, y_pred, y_prob)

    importance_table = pd.DataFrame({"feature": X_train_use.columns, "importance": model.feature_importances_})
    importance_table = importance_table.sort_values("importance", ascending=False).reset_index(drop=True)
    pred_df = pd.DataFrame({"y_true": y_valid_bin, "y_pred": y_pred, "y_prob": y_prob})

    summary = {
        "model_name": "optimized_random_forest_classifier",
        "fit_seconds": fit_seconds,
        "train_rows": int(len(X_train_use)),
        "valid_rows": int(len(X_valid_use)),
        "metrics": metrics,
        "tuned_with_optuna": bool(tune),
        "optuna_tuning_seconds": float(tune_report["tune_seconds"]) if tune_report is not None else 0.0,
        "optuna_trials_run": int(tune_report["n_trials_run"]) if tune_report is not None else 0,
        "optuna_best_inner_auc": float(tune_report["best_auc"]) if tune_report is not None else np.nan,
        "optuna_tune_rows": int(tune_report["tune_rows"]) if tune_report is not None else 0,
        "model_params": model_params,
        "prep_info": prep_info
    }

    if verbose:
        print("\nModel 4: Optimized random forest classifier")
        if tune:
            print(
                f"Optuna tuned: trials={summary['optuna_trials_run']} | "
                f"best inner AUC={summary['optuna_best_inner_auc']:.4f} | "
                f"tune_seconds={summary['optuna_tuning_seconds']:.2f} | "
                f"tune_rows={summary['optuna_tune_rows']:,}"
            )
        print(f"AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f}")

    return {
        "model": model,
        "summary": summary,
        "feature_importance": importance_table,
        "valid_predictions": pred_df
    }


def run_requested_model_suite(
        package,
        prep_report=None,
        seed=42,
        tune_lgbm=True,
        lgbm_optuna_trials=40,
        lgbm_optuna_timeout=None,
        tune_rf=True,
        rf_optuna_trials=30,
        rf_optuna_timeout=None,
        rf_max_tune_rows=150_000,
        verbose=True):
    X_train = package["X_train"]
    y_train = package["y_train"]
    X_valid = package["X_valid"]
    y_valid = package["y_valid"]
    target = package.get("target", "tip_amount")

    if prep_report is None:
        prep_report = run_model_prep_checks(
            X=X_train,
            y=y_train,
            target=target,
            top_n=10,
            corr_threshold=0.9,
            skew_threshold=1.0,
            show_plots=False
        )

    naive_reg = fit_naive_regression_baseline(y_train, y_valid, positive_only=True, verbose=verbose)
    naive_cls = fit_naive_classification_baseline(y_train, y_valid, verbose=verbose)

    simple_linear = fit_simple_linear_regression(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=prep_report,
        verbose=verbose
    )
    optimized_reg = fit_optimized_regression_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=prep_report,
        seed=seed,
        tune=tune_lgbm,
        optuna_trials=lgbm_optuna_trials,
        optuna_timeout=lgbm_optuna_timeout,
        verbose=verbose
    )
    simple_logistic = fit_simple_logistic_regression(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=prep_report,
        verbose=verbose
    )
    optimized_rf = fit_optimized_random_forest_classifier(
        X_train,
        y_train,
        X_valid,
        y_valid,
        prep_report=prep_report,
        seed=seed,
        tune=tune_rf,
        optuna_trials=rf_optuna_trials,
        optuna_timeout=rf_optuna_timeout,
        rf_max_tune_rows=rf_max_tune_rows,
        verbose=verbose
    )

    regression_table = pd.DataFrame([
        {
            "model": naive_reg["summary"]["model_name"],
            "rmse": naive_reg["summary"]["metrics"]["rmse"],
            "mae": naive_reg["summary"]["metrics"]["mae"],
            "r2": naive_reg["summary"]["metrics"]["r2"],
            "fit_seconds": naive_reg["summary"]["fit_seconds"]
        },
        {
            "model": simple_linear["summary"]["model_name"],
            "rmse": simple_linear["summary"]["metrics"]["rmse"],
            "mae": simple_linear["summary"]["metrics"]["mae"],
            "r2": simple_linear["summary"]["metrics"]["r2"],
            "fit_seconds": simple_linear["summary"]["fit_seconds"]
        },
        {
            "model": optimized_reg["summary"]["model_name"],
            "rmse": optimized_reg["summary"]["metrics"]["rmse"],
            "mae": optimized_reg["summary"]["metrics"]["mae"],
            "r2": optimized_reg["summary"]["metrics"]["r2"],
            "fit_seconds": optimized_reg["summary"]["fit_seconds"]
        }
    ]).sort_values("rmse").reset_index(drop=True)

    classification_table = pd.DataFrame([
        {
            "model": naive_cls["summary"]["model_name"],
            "roc_auc": naive_cls["summary"]["metrics"]["roc_auc"],
            "avg_precision": naive_cls["summary"]["metrics"]["avg_precision"],
            "f1": naive_cls["summary"]["metrics"]["f1"],
            "precision": naive_cls["summary"]["metrics"]["precision"],
            "recall": naive_cls["summary"]["metrics"]["recall"],
            "accuracy": naive_cls["summary"]["metrics"]["accuracy"],
            "fit_seconds": naive_cls["summary"]["fit_seconds"]
        },
        {
            "model": simple_logistic["summary"]["model_name"],
            "roc_auc": simple_logistic["summary"]["metrics"]["roc_auc"],
            "avg_precision": simple_logistic["summary"]["metrics"]["avg_precision"],
            "f1": simple_logistic["summary"]["metrics"]["f1"],
            "precision": simple_logistic["summary"]["metrics"]["precision"],
            "recall": simple_logistic["summary"]["metrics"]["recall"],
            "accuracy": simple_logistic["summary"]["metrics"]["accuracy"],
            "fit_seconds": simple_logistic["summary"]["fit_seconds"]
        },
        {
            "model": optimized_rf["summary"]["model_name"],
            "roc_auc": optimized_rf["summary"]["metrics"]["roc_auc"],
            "avg_precision": optimized_rf["summary"]["metrics"]["avg_precision"],
            "f1": optimized_rf["summary"]["metrics"]["f1"],
            "precision": optimized_rf["summary"]["metrics"]["precision"],
            "recall": optimized_rf["summary"]["metrics"]["recall"],
            "accuracy": optimized_rf["summary"]["metrics"]["accuracy"],
            "fit_seconds": optimized_rf["summary"]["fit_seconds"]
        }
    ]).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    if verbose:
        print("\nRegression model comparison")
        print(regression_table.round(4).to_string(index=False))
        print("\nClassification model comparison")
        print(classification_table.round(4).to_string(index=False))

    return {
        "naive_regression": naive_reg,
        "simple_linear": simple_linear,
        "optimized_regression": optimized_reg,
        "naive_classification": naive_cls,
        "simple_logistic": simple_logistic,
        "optimized_random_forest": optimized_rf,
        "regression_table": regression_table,
        "classification_table": classification_table
    }


# -----------------------------------------------------------------------------
# Interpretation report for interpretable models and optimized model importances
# -----------------------------------------------------------------------------
def build_interpretation_report(model_suite, top_n=12, verbose=True):
    simple_linear = model_suite["simple_linear"]
    simple_logistic = model_suite["simple_logistic"]
    optimized_regression = model_suite["optimized_regression"]
    optimized_rf = model_suite["optimized_random_forest"]

    linear_coef = simple_linear["coef_table"][["feature", "coef"]].copy()
    linear_positive = linear_coef.sort_values("coef", ascending=False).head(top_n).reset_index(drop=True)
    linear_negative = linear_coef.sort_values("coef", ascending=True).head(top_n).reset_index(drop=True)

    logistic_coef = simple_logistic["coef_table"][["feature", "log_odds_coef", "odds_ratio"]].copy()
    logistic_positive = logistic_coef.sort_values("log_odds_coef", ascending=False).head(top_n).reset_index(drop=True)
    logistic_negative = logistic_coef.sort_values("log_odds_coef", ascending=True).head(top_n).reset_index(drop=True)

    lgb_importance = optimized_regression["feature_importance"].copy()
    rf_importance = optimized_rf["feature_importance"].copy()
    lgb_top = lgb_importance.head(top_n).reset_index(drop=True)
    rf_top = rf_importance.head(top_n).reset_index(drop=True)

    logistic_intercept = simple_logistic["summary"]["intercept_log_odds"]
    logistic_base_prob = float(1.0 / (1.0 + np.exp(-logistic_intercept)))

    if verbose:
        print("\nInterpretation report")
        print("Simple linear regression interpretation")
        print("Each coefficient is the expected dollar change in tip_amount for a one-unit increase in that feature, holding others fixed")
        print(f"Intercept (expected tip at reference levels): {simple_linear['summary']['intercept']:.4f}")
        print("\nTop positive coefficients")
        print(linear_positive.round(4).to_string(index=False))
        print("\nTop negative coefficients")
        print(linear_negative.round(4).to_string(index=False))

        print("\nSimple logistic regression interpretation")
        print("log_odds_coef is additive on log-odds; odds_ratio > 1 increases odds of tipping, odds_ratio < 1 decreases odds")
        print(f"Intercept log-odds: {logistic_intercept:.4f} | baseline tip probability: {logistic_base_prob:.4f}")
        print("\nTop positive tipping drivers by log-odds")
        print(logistic_positive.round(4).to_string(index=False))
        print("\nTop negative tipping drivers by log-odds")
        print(logistic_negative.round(4).to_string(index=False))

        print("\nOptimized model feature importances")
        print("Top LightGBM regression importances")
        print(lgb_top.round(4).to_string(index=False))
        print("\nTop random forest classification importances")
        print(rf_top.round(4).to_string(index=False))

    return {
        "linear_positive": linear_positive,
        "linear_negative": linear_negative,
        "logistic_positive": logistic_positive,
        "logistic_negative": logistic_negative,
        "lgb_importance_top": lgb_top,
        "rf_importance_top": rf_top,
        "linear_intercept": float(simple_linear["summary"]["intercept"]),
        "logistic_intercept_log_odds": float(logistic_intercept),
        "logistic_baseline_tip_prob": logistic_base_prob
    }


def _ensure_year_artifact_dir(out_dir, year):
    year_dir = pathlib.Path(out_dir, f"year_{year}")
    year_dir.mkdir(parents=True, exist_ok=True)
    return year_dir


def _validate_crop_quantiles(crop_quantiles):
    if crop_quantiles is None:
        return
    if len(crop_quantiles) != 2:
        raise ValueError(f"crop_quantiles must have length 2, got {crop_quantiles}")
    low_q, high_q = crop_quantiles
    if not (0 <= low_q < high_q <= 1):
        raise ValueError(f"Expected 0 <= low_q < high_q <= 1, got {crop_quantiles}")


def _quantile_axis_limits(values, crop_quantiles=None, pad_frac=0.0):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return None
    if crop_quantiles is None:
        low = float(s.min())
        high = float(s.max())
    else:
        low_q, high_q = crop_quantiles
        low = float(s.quantile(low_q))
        high = float(s.quantile(high_q))
    span = high - low
    if span <= 0:
        pad = max(abs(high) * 0.05, 1e-6)
    else:
        pad = span * max(float(pad_frac), 0.0)
    return low - pad, high + pad


def _plot_single_regression_diagnostic(
        pred_df,
        model_name,
        out_path,
        max_points=REG_PLOT_MAX_POINTS,
        show_plot=False,
        crop_quantiles=DEFAULT_REG_PLOT_CROP_QUANTILES,
        crop_padding_frac=DEFAULT_REG_PLOT_CROP_PADDING_FRAC):
    needed_cols = {"y_true", "y_pred", "residual"}
    if not needed_cols.issubset(set(pred_df.columns)):
        raise KeyError(f"{model_name} predictions must include columns: {sorted(needed_cols)}")
    _validate_crop_quantiles(crop_quantiles)

    plot_df = pred_df.copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(data=plot_df, x="y_true", y="y_pred", s=12, alpha=0.35, ax=axes[0])
    combined = pd.concat([plot_df["y_true"], plot_df["y_pred"]], ignore_index=True)
    combined_limits = _quantile_axis_limits(combined, crop_quantiles=crop_quantiles, pad_frac=crop_padding_frac)
    if combined_limits is None:
        lower = float(min(plot_df["y_true"].min(), plot_df["y_pred"].min()))
        upper = float(max(plot_df["y_true"].max(), plot_df["y_pred"].max()))
    else:
        lower, upper = combined_limits
    axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_title("Actual vs Predicted")
    axes[0].set_xlabel("Actual tip")
    axes[0].set_ylabel("Predicted tip")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    sns.scatterplot(data=plot_df, x="y_pred", y="residual", s=12, alpha=0.35, ax=axes[1])
    pred_limits = _quantile_axis_limits(plot_df["y_pred"], crop_quantiles=crop_quantiles, pad_frac=crop_padding_frac)
    resid_limits = _quantile_axis_limits(plot_df["residual"], crop_quantiles=crop_quantiles, pad_frac=crop_padding_frac)
    if pred_limits is not None:
        axes[1].set_xlim(pred_limits[0], pred_limits[1])
    if resid_limits is not None:
        axes[1].set_ylim(resid_limits[0], resid_limits[1])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residual vs Predicted")
    axes[1].set_xlabel("Predicted tip")
    axes[1].set_ylabel("Residual")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    sns.histplot(plot_df["residual"], bins=70, kde=True, color="steelblue", ax=axes[2])
    if resid_limits is not None:
        axes[2].set_xlim(resid_limits[0], resid_limits[1])
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Residual Distribution")
    axes[2].set_xlabel("Residual")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(model_name, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path.as_posix(), dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_regression_diagnostic_plots(
        model_suite,
        year,
        out_dir=ARTIFACT_ROOT_DIR,
        show_plots=False,
        max_points=REG_PLOT_MAX_POINTS,
        crop_quantiles=DEFAULT_REG_PLOT_CROP_QUANTILES,
        crop_padding_frac=DEFAULT_REG_PLOT_CROP_PADDING_FRAC):
    year_dir = _ensure_year_artifact_dir(out_dir, year)
    plot_paths = {}
    model_items = [
        ("naive_regression", "naive_mean_tip_regression"),
        ("simple_linear", "simple_linear_regression"),
        ("optimized_regression", "optimized_lightgbm_regression")
    ]

    for suite_key, label in model_items:
        if suite_key not in model_suite or "valid_predictions" not in model_suite[suite_key]:
            continue
        out_path = year_dir / f"{label}_diagnostics.png"
        _plot_single_regression_diagnostic(
            pred_df=model_suite[suite_key]["valid_predictions"],
            model_name=label,
            out_path=out_path,
            max_points=max_points,
            show_plot=show_plots,
            crop_quantiles=crop_quantiles,
            crop_padding_frac=crop_padding_frac
        )
        plot_paths[label] = out_path.as_posix()

    return plot_paths


def build_year_summary_row(year, package, model_suite):
    outlier_summary = package.get("outlier_summary", {})
    train_outlier = outlier_summary.get("train", {})
    valid_outlier = outlier_summary.get("valid", {})

    reg_table = model_suite["regression_table"]
    cls_table = model_suite["classification_table"]
    best_reg = reg_table.iloc[0]
    best_cls = cls_table.iloc[0]

    naive_reg_rmse = float(model_suite["naive_regression"]["summary"]["metrics"]["rmse"])
    naive_cls_auc = float(model_suite["naive_classification"]["summary"]["metrics"]["roc_auc"])
    best_reg_rmse = float(best_reg["rmse"])
    best_cls_auc = float(best_cls["roc_auc"])

    opt_reg_summary = model_suite["optimized_regression"]["summary"]
    opt_rf_summary = model_suite["optimized_random_forest"]["summary"]

    return {
        "year": int(year),
        "target": package.get("target", "tip_amount"),
        "sample_rows": int(len(package["X"])),
        "train_rows": int(len(package["X_train"])),
        "valid_rows": int(len(package["X_valid"])),
        "train_tip_share_pct": float((package["y_train"] > 0).mean() * 100),
        "valid_tip_share_pct": float((package["y_valid"] > 0).mean() * 100),
        "outlier_method": outlier_summary.get("method", "unknown"),
        "outlier_cols_used_n": int(len(outlier_summary.get("columns_used", []))),
        "outlier_train_rows_removed": int(train_outlier.get("rows_removed", 0)),
        "outlier_valid_rows_removed": int(valid_outlier.get("rows_removed", 0)),
        "best_reg_model": str(best_reg["model"]),
        "best_reg_rmse": best_reg_rmse,
        "best_reg_mae": float(best_reg["mae"]),
        "best_reg_r2": float(best_reg["r2"]),
        "naive_reg_rmse": naive_reg_rmse,
        "best_reg_rmse_gain_vs_naive": float(naive_reg_rmse - best_reg_rmse),
        "best_cls_model": str(best_cls["model"]),
        "best_cls_roc_auc": best_cls_auc,
        "best_cls_avg_precision": float(best_cls["avg_precision"]),
        "best_cls_f1": float(best_cls["f1"]),
        "naive_cls_roc_auc": naive_cls_auc,
        "best_cls_auc_gain_vs_naive": float(best_cls_auc - naive_cls_auc),
        "opt_lgbm_tuned": bool(opt_reg_summary.get("tuned_with_optuna", False)),
        "opt_lgbm_trials": int(opt_reg_summary.get("optuna_trials_run", 0)),
        "opt_rf_tuned": bool(opt_rf_summary.get("tuned_with_optuna", False)),
        "opt_rf_trials": int(opt_rf_summary.get("optuna_trials_run", 0))
    }


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, pathlib.Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _format_feature_markdown(df, value_col, top_n=5, descending=True):
    if df is None or len(df) == 0:
        return ["- none"]
    if "feature" not in df.columns or value_col not in df.columns:
        return ["- none"]

    work = df[["feature", value_col]].copy().dropna()
    if len(work) == 0:
        return ["- none"]

    work = work.sort_values(value_col, ascending=not descending).head(top_n)
    rows = []
    for _, row in work.iterrows():
        rows.append(f"- `{row['feature']}`: {float(row[value_col]):.4f}")
    return rows if len(rows) > 0 else ["- none"]


def build_run_manifest(
        year,
        package,
        model_suite,
        prep_report=None,
        run_config=None,
        export_config=None,
        csv_paths=None,
        plot_paths=None):
    reg_table = model_suite["regression_table"].copy()
    cls_table = model_suite["classification_table"].copy()
    best_reg = reg_table.iloc[0].to_dict() if len(reg_table) > 0 else {}
    best_cls = cls_table.iloc[0].to_dict() if len(cls_table) > 0 else {}
    outlier_summary = package.get("outlier_summary", {})
    opt_reg_summary = model_suite["optimized_regression"]["summary"]
    opt_rf_summary = model_suite["optimized_random_forest"]["summary"]

    prep_summary = {}
    if prep_report is not None:
        groups = prep_report.get("feature_groups", {})
        scaling = prep_report.get("scaling", {})
        prep_summary = {
            "continuous_n": int(len(groups.get("continuous_cols", []))),
            "low_card_n": int(len(groups.get("low_card_cols", []))),
            "binary_n": int(len(groups.get("binary_cols", []))),
            "dummy_n": int(len(groups.get("dummy_cols", []))),
            "high_corr_pairs_n": int(len(prep_report.get("high_corr_pairs", []))),
            "zero_var_n": int(len(prep_report.get("low_variance", {}).get("zero_var_features", []))),
            "near_zero_var_n": int(len(prep_report.get("low_variance", {}).get("near_zero_var_features", []))),
            "recommend_scaling": bool(scaling.get("recommend_scaling", False)),
            "recommended_scaler": scaling.get("recommended_scaler", "unknown")
        }

    manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "year": int(year),
        "target": package.get("target", "tip_amount"),
        "row_counts": {
            "sample_rows": int(len(package["X"])),
            "train_rows": int(len(package["X_train"])),
            "valid_rows": int(len(package["X_valid"]))
        },
        "tip_share_pct": {
            "train": float((package["y_train"] > 0).mean() * 100),
            "valid": float((package["y_valid"] > 0).mean() * 100)
        },
        "outlier_summary": outlier_summary,
        "prep_summary": prep_summary,
        "best_models": {
            "regression": best_reg,
            "classification": best_cls
        },
        "model_tables": {
            "regression": reg_table.to_dict(orient="records"),
            "classification": cls_table.to_dict(orient="records")
        },
        "tuning": {
            "lightgbm": {
                "enabled": bool(opt_reg_summary.get("tuned_with_optuna", False)),
                "trials_run": int(opt_reg_summary.get("optuna_trials_run", 0)),
                "tuning_seconds": float(opt_reg_summary.get("optuna_tuning_seconds", 0.0)),
                "best_inner_rmse": float(opt_reg_summary.get("optuna_best_inner_rmse", np.nan))
            },
            "random_forest": {
                "enabled": bool(opt_rf_summary.get("tuned_with_optuna", False)),
                "trials_run": int(opt_rf_summary.get("optuna_trials_run", 0)),
                "tuning_seconds": float(opt_rf_summary.get("optuna_tuning_seconds", 0.0)),
                "best_inner_auc": float(opt_rf_summary.get("optuna_best_inner_auc", np.nan)),
                "tune_rows": int(opt_rf_summary.get("optuna_tune_rows", 0))
            }
        },
        "run_config": run_config if run_config is not None else {},
        "export_config": export_config if export_config is not None else {},
        "artifact_files": {
            "csv": csv_paths if csv_paths is not None else {},
            "plots": plot_paths if plot_paths is not None else {}
        },
        "library_versions": {
            "python": sys.version.split(" ")[0],
            "duckdb": duckdb.__version__,
            "optuna": optuna.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__, # type: ignore
            "seaborn": sns.__version__ # type: ignore
        }
    }
    return _to_builtin(manifest)


def build_year_report_markdown(year, package, model_suite, interpretation_report=None, run_config=None):
    summary = build_year_summary_row(year=year, package=package, model_suite=model_suite)
    outlier_summary = package.get("outlier_summary", {})
    outlier_cols = outlier_summary.get("columns_used", [])
    train_outlier = outlier_summary.get("train", {})
    valid_outlier = outlier_summary.get("valid", {})

    lines = [
        f"# NYC Yellow Taxi Modeling Report ({year})",
        "",
        "## Scope",
        f"- Target: `{summary['target']}`",
        f"- Sample rows: {summary['sample_rows']:,}",
        f"- Train rows: {summary['train_rows']:,}",
        f"- Validation rows: {summary['valid_rows']:,}",
        f"- Train tip share: {summary['train_tip_share_pct']:.2f}%",
        f"- Validation tip share: {summary['valid_tip_share_pct']:.2f}%",
        "",
        "## Best Models",
        f"- Regression winner: `{summary['best_reg_model']}` | RMSE={summary['best_reg_rmse']:.4f} | MAE={summary['best_reg_mae']:.4f} | R2={summary['best_reg_r2']:.4f}",
        f"- Classification winner: `{summary['best_cls_model']}` | ROC-AUC={summary['best_cls_roc_auc']:.4f} | AP={summary['best_cls_avg_precision']:.4f} | F1={summary['best_cls_f1']:.4f}",
        "",
        "## Baseline Gains",
        f"- Regression RMSE gain vs naive mean: {summary['best_reg_rmse_gain_vs_naive']:.4f}",
        f"- Classification AUC gain vs naive majority: {summary['best_cls_auc_gain_vs_naive']:.4f}",
        "",
        "## Cleaning Summary",
        f"- Outlier method: `{summary['outlier_method']}`",
        f"- Outlier columns used ({len(outlier_cols)}): {outlier_cols}",
        f"- Train rows removed by outlier step: {int(train_outlier.get('rows_removed', 0)):,}",
        f"- Validation rows removed by outlier step: {int(valid_outlier.get('rows_removed', 0)):,}"
    ]

    if interpretation_report is not None:
        lines.extend([
            "",
            "## Interpretability Highlights",
            "### Simple Linear Regression (Top Positive Coefficients)"
        ])
        lines.extend(_format_feature_markdown(interpretation_report.get("linear_positive"), value_col="coef", top_n=5, descending=True))
        lines.extend([
            "",
            "### Simple Logistic Regression (Top Positive Log-Odds Drivers)"
        ])
        lines.extend(_format_feature_markdown(interpretation_report.get("logistic_positive"), value_col="log_odds_coef", top_n=5, descending=True))

    if run_config is not None and len(run_config) > 0:
        lines.extend([
            "",
            "## Run Configuration",
            "```json",
            json.dumps(_to_builtin(run_config), indent=2),
            "```"
        ])

    lines.extend([
        "",
        "## Caveats",
        "- Results are observational and support predictive guidance, not causal claims",
        "- Performance may vary with macro shifts and policy changes across years",
        "- Review subgroup errors before using model outputs for operational recommendations"
    ])
    return "\n".join(lines)


def export_year_artifacts(
        year,
        package,
        model_suite,
        interpretation_report=None,
        prep_report=None,
        run_config=None,
        out_dir=ARTIFACT_ROOT_DIR,
        save_plots=True,
        show_plots=False,
        plot_crop_quantiles=DEFAULT_REG_PLOT_CROP_QUANTILES,
        plot_crop_padding_frac=DEFAULT_REG_PLOT_CROP_PADDING_FRAC,
        write_manifest=True,
        write_year_report=True,
        verbose=True):
    year_dir = _ensure_year_artifact_dir(out_dir, year)
    csv_paths = {}

    reg_cmp_path = year_dir / "regression_model_comparison.csv"
    model_suite["regression_table"].to_csv(reg_cmp_path.as_posix(), index=False)
    csv_paths["regression_model_comparison"] = reg_cmp_path.as_posix()

    cls_cmp_path = year_dir / "classification_model_comparison.csv"
    model_suite["classification_table"].to_csv(cls_cmp_path.as_posix(), index=False)
    csv_paths["classification_model_comparison"] = cls_cmp_path.as_posix()

    baseline_rows = [
        {
            "model": model_suite["naive_regression"]["summary"]["model_name"],
            "fit_seconds": model_suite["naive_regression"]["summary"]["fit_seconds"],
            "rmse": model_suite["naive_regression"]["summary"]["metrics"]["rmse"],
            "mae": model_suite["naive_regression"]["summary"]["metrics"]["mae"],
            "r2": model_suite["naive_regression"]["summary"]["metrics"]["r2"]
        },
        {
            "model": model_suite["naive_classification"]["summary"]["model_name"],
            "fit_seconds": model_suite["naive_classification"]["summary"]["fit_seconds"],
            "roc_auc": model_suite["naive_classification"]["summary"]["metrics"]["roc_auc"],
            "avg_precision": model_suite["naive_classification"]["summary"]["metrics"]["avg_precision"],
            "f1": model_suite["naive_classification"]["summary"]["metrics"]["f1"],
            "precision": model_suite["naive_classification"]["summary"]["metrics"]["precision"],
            "recall": model_suite["naive_classification"]["summary"]["metrics"]["recall"],
            "accuracy": model_suite["naive_classification"]["summary"]["metrics"]["accuracy"]
        }
    ]
    baseline_path = year_dir / "baseline_model_summary.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_path.as_posix(), index=False)
    csv_paths["baseline_model_summary"] = baseline_path.as_posix()

    simple_coef_path = year_dir / "simple_linear_coefficients.csv"
    model_suite["simple_linear"]["coef_table"].to_csv(simple_coef_path.as_posix(), index=False)
    csv_paths["simple_linear_coefficients"] = simple_coef_path.as_posix()

    simple_logit_coef_path = year_dir / "simple_logistic_coefficients.csv"
    model_suite["simple_logistic"]["coef_table"].to_csv(simple_logit_coef_path.as_posix(), index=False)
    csv_paths["simple_logistic_coefficients"] = simple_logit_coef_path.as_posix()

    lgbm_imp_path = year_dir / "optimized_lgbm_feature_importance.csv"
    model_suite["optimized_regression"]["feature_importance"].to_csv(lgbm_imp_path.as_posix(), index=False)
    csv_paths["optimized_lgbm_feature_importance"] = lgbm_imp_path.as_posix()

    rf_imp_path = year_dir / "optimized_rf_feature_importance.csv"
    model_suite["optimized_random_forest"]["feature_importance"].to_csv(rf_imp_path.as_posix(), index=False)
    csv_paths["optimized_rf_feature_importance"] = rf_imp_path.as_posix()

    prediction_exports = {
        "naive_regression_predictions": model_suite["naive_regression"]["valid_predictions"],
        "simple_linear_predictions": model_suite["simple_linear"]["valid_predictions"],
        "optimized_lgbm_predictions": model_suite["optimized_regression"]["valid_predictions"],
        "naive_classification_predictions": model_suite["naive_classification"]["valid_predictions"],
        "simple_logistic_predictions": model_suite["simple_logistic"]["valid_predictions"],
        "optimized_rf_predictions": model_suite["optimized_random_forest"]["valid_predictions"]
    }
    for name, df in prediction_exports.items():
        out_path = year_dir / f"{name}.csv"
        df.to_csv(out_path.as_posix(), index=False)
        csv_paths[name] = out_path.as_posix()

    if interpretation_report is not None:
        interp_paths = {
            "interpretation_linear_positive": interpretation_report["linear_positive"],
            "interpretation_linear_negative": interpretation_report["linear_negative"],
            "interpretation_logistic_positive": interpretation_report["logistic_positive"],
            "interpretation_logistic_negative": interpretation_report["logistic_negative"],
            "interpretation_lgb_importance_top": interpretation_report["lgb_importance_top"],
            "interpretation_rf_importance_top": interpretation_report["rf_importance_top"]
        }
        for name, df in interp_paths.items():
            out_path = year_dir / f"{name}.csv"
            df.to_csv(out_path.as_posix(), index=False)
            csv_paths[name] = out_path.as_posix()

    outlier_summary_path = year_dir / "outlier_summary_flat.csv"
    pd.json_normalize(package.get("outlier_summary", {}), sep="_").to_csv(outlier_summary_path.as_posix(), index=False)
    csv_paths["outlier_summary_flat"] = outlier_summary_path.as_posix()

    year_summary = build_year_summary_row(year=year, package=package, model_suite=model_suite)
    year_summary_path = year_dir / "year_summary.csv"
    pd.DataFrame([year_summary]).to_csv(year_summary_path.as_posix(), index=False)
    csv_paths["year_summary"] = year_summary_path.as_posix()

    plot_paths = {}
    if save_plots:
        plot_paths = save_regression_diagnostic_plots(
            model_suite=model_suite,
            year=year,
            out_dir=out_dir,
            show_plots=show_plots,
            max_points=REG_PLOT_MAX_POINTS,
            crop_quantiles=plot_crop_quantiles,
            crop_padding_frac=plot_crop_padding_frac
        )

    export_config = {
        "save_plots": bool(save_plots),
        "show_plots": bool(show_plots),
        "plot_crop_quantiles": plot_crop_quantiles,
        "plot_crop_padding_frac": float(plot_crop_padding_frac),
        "write_manifest": bool(write_manifest),
        "write_year_report": bool(write_year_report)
    }

    manifest = None
    manifest_path = None
    if write_manifest:
        manifest = build_run_manifest(
            year=year,
            package=package,
            model_suite=model_suite,
            prep_report=prep_report,
            run_config=run_config,
            export_config=export_config,
            csv_paths=csv_paths,
            plot_paths=plot_paths
        )
        manifest_path = year_dir / MANIFEST_FILE_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    year_report_path = None
    if write_year_report:
        report_md = build_year_report_markdown(
            year=year,
            package=package,
            model_suite=model_suite,
            interpretation_report=interpretation_report,
            run_config=run_config
        )
        year_report_path = year_dir / YEAR_REPORT_FILE_NAME
        year_report_path.write_text(report_md, encoding="utf-8")

    if verbose:
        print(f"\nSaved year {year} artifacts -> {year_dir.as_posix()}")
        print(f"CSV files saved: {len(csv_paths)} | Plot files saved: {len(plot_paths)} | Manifest: {bool(write_manifest)} | Report: {bool(write_year_report)}")

    return {
        "year_dir": year_dir.as_posix(),
        "csv_paths": csv_paths,
        "plot_paths": plot_paths,
        "year_summary": year_summary,
        "manifest_path": manifest_path.as_posix() if manifest_path is not None else None,
        "year_report_path": year_report_path.as_posix() if year_report_path is not None else None
    }


def run_yearly_modeling_exports(
        years=VALID_YEARS,
        target="tip_amount",
        target_rows=250_000,
        seed=42,
        out_dir=ARTIFACT_ROOT_DIR,
        run_variable_checks=False,
        outlier_method="winsor",
        outlier_lower=0.003,
        outlier_upper=0.997,
        valid_frac=0.2,
        tune_lgbm=True,
        lgbm_optuna_trials=80,
        lgbm_optuna_timeout=None,
        tune_rf=True,
        rf_optuna_trials=20,
        rf_optuna_timeout=None,
        rf_max_tune_rows=150_000,
        plot_crop_quantiles=DEFAULT_REG_PLOT_CROP_QUANTILES,
        plot_crop_padding_frac=DEFAULT_REG_PLOT_CROP_PADDING_FRAC,
        write_manifest=True,
        write_year_report=True,
        verbose=True):
    years = sorted(years)
    summary_rows = []

    for year in years:
        _validate_year(year)
        if verbose:
            print("\n" + "=" * 72)
            print(f"Running modeling export for year {year}")
            print("=" * 72)

        run_config = {
            "year": int(year),
            "target": target,
            "target_rows": int(target_rows),
            "seed": int(seed),
            "valid_frac": float(valid_frac),
            "outlier_method": outlier_method,
            "outlier_lower": float(outlier_lower),
            "outlier_upper": float(outlier_upper),
            "tune_lgbm": bool(tune_lgbm),
            "lgbm_optuna_trials": int(lgbm_optuna_trials),
            "lgbm_optuna_timeout": lgbm_optuna_timeout,
            "tune_rf": bool(tune_rf),
            "rf_optuna_trials": int(rf_optuna_trials),
            "rf_optuna_timeout": rf_optuna_timeout,
            "rf_max_tune_rows": int(rf_max_tune_rows),
            "plot_crop_quantiles": plot_crop_quantiles,
            "plot_crop_padding_frac": float(plot_crop_padding_frac)
        }

        package = build_eda_dataset(
            year=year,
            target=target,
            target_rows=target_rows,
            seed=seed,
            run_variable_checks=run_variable_checks,
            outlier_method=outlier_method,
            outlier_lower=outlier_lower,
            outlier_upper=outlier_upper,
            valid_frac=valid_frac,
            show_plots=False
        )
        prep_report = run_model_prep_checks(
            X=package["X_train"],
            y=package["y_train"],
            target=target,
            top_n=12,
            corr_threshold=0.9,
            skew_threshold=1.0,
            show_plots=False
        )
        model_suite = run_requested_model_suite(
            package=package,
            prep_report=prep_report,
            seed=seed,
            tune_lgbm=tune_lgbm,
            lgbm_optuna_trials=lgbm_optuna_trials,
            lgbm_optuna_timeout=lgbm_optuna_timeout,
            tune_rf=tune_rf,
            rf_optuna_trials=rf_optuna_trials,
            rf_optuna_timeout=rf_optuna_timeout,
            rf_max_tune_rows=rf_max_tune_rows,
            verbose=verbose
        )
        interpretation_report = build_interpretation_report(model_suite, top_n=10, verbose=False)
        export_result = export_year_artifacts(
            year=year,
            package=package,
            model_suite=model_suite,
            interpretation_report=interpretation_report,
            prep_report=prep_report,
            run_config=run_config,
            out_dir=out_dir,
            save_plots=True,
            show_plots=False,
            plot_crop_quantiles=plot_crop_quantiles,
            plot_crop_padding_frac=plot_crop_padding_frac,
            write_manifest=write_manifest,
            write_year_report=write_year_report,
            verbose=verbose
        )
        summary_rows.append(export_result["year_summary"])

    all_summary = pd.DataFrame(summary_rows).sort_values("year").reset_index(drop=True)
    out_root = pathlib.Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_years_path = out_root / "all_years_summary.csv"
    all_summary.to_csv(all_years_path.as_posix(), index=False)
    if verbose:
        print(f"\nSaved multi-year summary -> {all_years_path.as_posix()}")
    return all_summary


# -----------------------------------------------------------------------------
# End-to-end EDA + modeling prep pipeline
# -----------------------------------------------------------------------------
def build_eda_dataset(
        year,
        target="tip_amount",
        target_rows=250_000,
        seed=42,
        run_variable_checks=False,
        check_cols=DEFAULT_PROFILE_COLS,
        outlier_cols=DEFAULT_OUTLIER_COLS,
        outlier_method="winsor",
        outlier_lower=0.003,
        outlier_upper=0.997,
        valid_frac=0.2,
        show_plots=True):
    _validate_target(target)
    if not (0 < valid_frac < 1):
        raise ValueError(f"valid_frac must be in (0,1), got {valid_frac}")

    # 1) Sample from cleaned parquet
    regr_df = fetch_regr_year(year=year, target_rows=target_rows, seed=seed, verbose=True)
    print_data_health(regr_df, title=f"Sample QA ({year})")

    # 2) Optional per-variable diagnostics before cleaning
    if run_variable_checks:
        for c in check_cols:
            full_variable_check(regr_df, c, show_plots=show_plots)

    # 3) Categorical encoding and transform creation
    model_df, new_features = encode_model_features(regr_df, verbose=True)

    # 4) Leakage-safe model matrix by selected target
    X, y = build_model_matrix(model_df, target=target, verbose=True)

    # 5) Reproducible train/validation split stratified on tip/no-tip class
    tip_class = (y > 0).astype(int)
    stratify_labels = tip_class if tip_class.nunique() == 2 else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_frac,
        random_state=seed,
        stratify=stratify_labels
    )
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    print(f"Train size: {len(X_train):,} | Valid size: {len(X_valid):,}")
    if stratify_labels is not None:
        train_tip_pct = float((y_train > 0).mean() * 100)
        valid_tip_pct = float((y_valid > 0).mean() * 100)
        print(f"Stratified tip share -> train: {train_tip_pct:.2f}% | valid: {valid_tip_pct:.2f}%")

    # 6) Fit outlier bounds on train only, then apply to train and valid
    outlier_cols_in_X = [c for c in outlier_cols if c in X_train.columns]
    if len(outlier_cols_in_X) == 0:
        outlier_bounds = {}
        train_outlier_summary = {
            "method": outlier_method,
            "rows_before": int(len(X_train)),
            "rows_after": int(len(X_train)),
            "rows_removed": 0,
            "rows_removed_pct": 0.0,
            "cells_clipped": 0
        }
        valid_outlier_summary = {
            "method": outlier_method,
            "rows_before": int(len(X_valid)),
            "rows_after": int(len(X_valid)),
            "rows_removed": 0,
            "rows_removed_pct": 0.0,
            "cells_clipped": 0
        }
        outlier_summary = {
            "method": outlier_method,
            "columns_used": [],
            "train": train_outlier_summary,
            "valid": valid_outlier_summary
        }
    else:
        print("\nFitting outlier bounds on train fold only")
        outlier_bounds = fit_outlier_bounds(
            X_train,
            cols=outlier_cols_in_X,
            lower=outlier_lower,
            upper=outlier_upper,
            verbose=True
        )
        X_train, y_train, train_outlier_summary = apply_outlier_bounds_to_xy(
            X_train,
            y_train,
            outlier_bounds,
            method=outlier_method,
            verbose=True,
            label="train"
        )
        X_valid, y_valid, valid_outlier_summary = apply_outlier_bounds_to_xy(
            X_valid,
            y_valid,
            outlier_bounds,
            method=outlier_method,
            verbose=True,
            label="valid"
        )
        outlier_summary = {
            "method": outlier_method,
            "columns_used": outlier_cols_in_X,
            "train": train_outlier_summary,
            "valid": valid_outlier_summary
        }

    print(f"Train size after outlier step: {len(X_train):,} | Valid size after outlier step: {len(X_valid):,}")

    return {
        "target": target,
        "full_df": model_df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "engineered_features": new_features,
        "outlier_bounds": outlier_bounds,
        "outlier_summary": outlier_summary
    }


if __name__ == "__main__":
    analysis_year = 2015
    target = "tip_amount"
    seed = 42
    target_rows = 250_000
    outlier_method = "winsor"
    lgbm_optuna_trials = 100
    rf_optuna_trials = 30
    package = build_eda_dataset(
        year=analysis_year,
        target=target,
        target_rows=target_rows,
        seed=seed,
        run_variable_checks=False,
        outlier_method=outlier_method,
        show_plots=False
    )

    print("\nFirst 5 rows of final model dataframe:")
    print(package["full_df"].head().to_string(index=False))
    print("\nEngineered features:")
    print(package["engineered_features"])
    print("\nOutlier summary:")
    print(package["outlier_summary"])

    # Run diagnostics on train fold only to avoid validation leakage in prep decisions
    prep_report = run_model_prep_checks(
        X=package["X_train"],
        y=package["y_train"],
        target=target,
        top_n=12,
        corr_threshold=0.9,
        skew_threshold=1.0,
        show_plots=False
    )

    print("\nPrep summary keys:")
    print(list(prep_report.keys()))

    model_suite = run_requested_model_suite(
        package=package,
        prep_report=prep_report,
        seed=seed,
        tune_lgbm=True,
        lgbm_optuna_trials=lgbm_optuna_trials,
        lgbm_optuna_timeout=None,
        tune_rf=True,
        rf_optuna_trials=rf_optuna_trials,
        rf_optuna_timeout=None,
        rf_max_tune_rows=150_000,
        verbose=True
    )
    interpretation_report = build_interpretation_report(model_suite, top_n=10, verbose=True)
    run_config = {
        "year": int(analysis_year),
        "target": target,
        "target_rows": int(target_rows),
        "seed": int(seed),
        "outlier_method": outlier_method,
        "tune_lgbm": True,
        "lgbm_optuna_trials": int(lgbm_optuna_trials),
        "tune_rf": True,
        "rf_optuna_trials": int(rf_optuna_trials),
        "plot_crop_quantiles": DEFAULT_REG_PLOT_CROP_QUANTILES,
        "plot_crop_padding_frac": float(DEFAULT_REG_PLOT_CROP_PADDING_FRAC)
    }
    artifact_result = export_year_artifacts(
        year=analysis_year,
        package=package,
        model_suite=model_suite,
        interpretation_report=interpretation_report,
        prep_report=prep_report,
        run_config=run_config,
        out_dir=ARTIFACT_ROOT_DIR,
        save_plots=True,
        show_plots=False,
        plot_crop_quantiles=DEFAULT_REG_PLOT_CROP_QUANTILES,
        plot_crop_padding_frac=DEFAULT_REG_PLOT_CROP_PADDING_FRAC,
        write_manifest=True,
        write_year_report=True,
        verbose=True
    )

    best_reg = model_suite["regression_table"].iloc[0]
    best_cls = model_suite["classification_table"].iloc[0]

    print("\nBest regression model (by RMSE):")
    print(f"{best_reg['model']} | RMSE={best_reg['rmse']:.4f} | MAE={best_reg['mae']:.4f} | R2={best_reg['r2']:.4f}")
    print("\nBest classification model (by AUC):")
    print(
        f"{best_cls['model']} | AUC={best_cls['roc_auc']:.4f} | F1={best_cls['f1']:.4f} | "
        f"Precision={best_cls['precision']:.4f} | Recall={best_cls['recall']:.4f}"
    )
    print("\nInterpretation summary keys:")
    print(list(interpretation_report.keys()))
    print("\nArtifact output directory:")
    print(artifact_result["year_dir"])
