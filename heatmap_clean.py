import duckdb, os, pathlib
import pandas as pd

ZONES_SHAPE = "./taxi_zones_4326.parquet"
PARQUET_DIR = "./taxi_parquets"


def build_heatmap_year(year, zones_path, out_dir):

    if year < 2015 or year > 2022:
        raise ValueError("year must be in 2015..2022")

    parquet_path = pathlib.Path(PARQUET_DIR,   f"yellow_clean_{year}.parquet").as_posix()

    con = duckdb.connect()
    try:
        con.execute("SET enable_object_cache = true;")
        con.execute("SET threads = 4;")

        # Keep heavy aggregation in DuckDB for memory-safe yearly processing
        sql = f"""
    PRAGMA disable_progress_bar;

    WITH zones AS (
      SELECT
        CAST(zone_id AS INT)    AS zone_id,
        CAST(borough AS VARCHAR) AS borough,
        CAST(zone_name AS VARCHAR) AS zone_name,
        CAST(wkt AS VARCHAR)    AS wkt
      FROM read_parquet('{zones_path}')
    ),

    -- Trip-level base rates with a minimum-fare guardrail for stable tip rate stats
    rates_base AS (
      SELECT
        pickup_location_id       AS zone_id,
        tip_amount::DOUBLE       AS tip_amount,
        fare_amount::DOUBLE      AS fare_amount,
        CASE WHEN fare_amount >= 5 THEN tip_amount / fare_amount END AS tip_rate_for_stats
      FROM read_parquet('{parquet_path}')
      WHERE pickup_location_id NOT IN (264, 265)
    ),

    -- Per-zone winsor bounds so outlier clipping is local to each pickup zone
    q AS (
      SELECT
        zone_id,
        quantile_cont(tip_rate_for_stats, 0.01) AS q01,
        quantile_cont(tip_rate_for_stats, 0.99) AS q99
      FROM rates_base
      WHERE tip_rate_for_stats IS NOT NULL
      GROUP BY zone_id
    ),

    -- Clip extreme tip rates to stabilize means without dropping trips
    rates_clipped AS (
      SELECT
        r.zone_id,
        r.tip_amount,
        r.fare_amount,
        r.tip_rate_for_stats                                     AS tip_rate_raw,
        CASE
          WHEN r.tip_rate_for_stats IS NULL THEN NULL
          WHEN r.tip_rate_for_stats < q.q01 THEN q.q01
          WHEN r.tip_rate_for_stats > q.q99 THEN q.q99
          ELSE r.tip_rate_for_stats
        END                                                     AS tip_rate_w
      FROM rates_base r
      LEFT JOIN q USING (zone_id)
    ),

    -- Keep zones with enough volume or acceptable relative sampling error
    agg AS (
      SELECT zone_id, avg_tip, med_tip, avg_fare, avg_rate, avg_rate_revenue, num_trips
      FROM (
        SELECT
          zone_id,
          AVG(tip_amount)                               AS avg_tip,
          median(tip_amount)                            AS med_tip,
          AVG(fare_amount)                              AS avg_fare,
          (AVG(tip_rate_w) * 100.0)                     AS avg_rate,
          (SUM(tip_amount) / SUM(fare_amount)) * 100.0  AS avg_rate_revenue,
          COUNT(*)                                      AS num_trips,
          quantile_cont(tip_amount, 0.75) - quantile_cont(tip_amount, 0.25) AS iqr
        FROM rates_clipped
        GROUP BY zone_id
      )
      -- Relative SE approximation: (sigma / sqrt(n)) / median_tip <= 15%
      WHERE (num_trips >= 100
        OR (((iqr / 1.349) / sqrt(num_trips)) / GREATEST(med_tip, 1.0) <= 0.15 AND num_trips >= 50))
    )

    SELECT
      z.zone_id,
      z.borough,
      z.zone_name,
      z.wkt,
      a.avg_tip,
      a.med_tip,
      a.avg_fare,
      a.avg_rate,
      a.avg_rate_revenue,
      a.num_trips
    FROM zones z
    LEFT JOIN agg a ON a.zone_id = z.zone_id
    ORDER BY a.num_trips DESC NULLS LAST, z.zone_id;
    """

        df = con.execute(sql).fetch_df()
    finally:
        con.close()


    # Write per-year CSV with WKT retained for downstream map joins
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heat_{year}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{year}] wrote {out_path}  ({len(df)} zones)")
    return df

if __name__ == "__main__":
    dfs = {}
    for y in range(2015, 2023):
        dfs[y] = build_heatmap_year(y, zones_path=ZONES_SHAPE, out_dir="./heatmaps")
