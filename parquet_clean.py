import duckdb, os, pathlib

# Paths
YEARLY_DIR = "./taxi_parquets"
ZONES_PATH  = "./taxi_zones_4326.parquet"
OUT_DIR    = "./taxi_parquets"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = range(2015, 2023)

# Constants for meter logic
INITIAL_DEFAULT = 2.50
UNIT_RATE_DEFAULT = 0.50
INITIAL_2022_NEW  = 3.00
UNIT_2022_NEW     = 0.70
MI_PER_TICK       = 0.2
SEC_PER_TICK      = 60.0
JFK_FLAT_DEFAULT  = 52.0
JFK_FLAT_2022     = 70.0
CUTOVER_2022      = "2022-12-19"
EXTRA_CAP         = 5.0
MTA_TAX_CAP       = 1.0
IMPROVEMENT_CAP   = 1.0
CONGESTION_CAP    = 5.0
AIRPORT_FEE_CAP   = 2.0

PRAGMAS = [
    "PRAGMA threads=4",
    "PRAGMA memory_limit='2GB'",
    "PRAGMA preserve_insertion_order=false"
]

SQL_TEMPLATE = """
{pragmas}
-- Write cleaned/enriched year in a streaming way
COPY (
  WITH src AS (
    SELECT
      tpep_pickup_datetime  AS pickup_datetime,
      tpep_dropoff_datetime AS dropoff_datetime,
      CAST(passenger_count AS DOUBLE) AS passenger_count,
      CAST(trip_distance  AS DOUBLE)  AS trip_distance,
      CAST(CAST(RatecodeID AS NUMERIC) AS INT) AS rate_code,
      CAST(PULocationID   AS INT)     AS pickup_location_id,
      CAST(DOLocationID   AS INT)     AS dropoff_location_id,
      CAST(fare_amount    AS DOUBLE)  AS fare_amount,
      CAST(extra          AS DOUBLE)  AS extra,
      CAST(mta_tax        AS DOUBLE)  AS mta_tax,
      CAST(improvement_surcharge AS DOUBLE) AS improvement_surcharge,
      CAST(congestion_surcharge  AS DOUBLE) AS congestion_surcharge,
      CAST(airport_fee           AS DOUBLE) AS airport_fee,
      CAST(tip_amount     AS DOUBLE)  AS tip_amount,
      CAST(tolls_amount   AS DOUBLE)  AS tolls_amount,
      CAST(total_amount   AS DOUBLE)  AS total_amount,
      CAST(payment_type   AS INT)     AS payment_type
    FROM read_parquet('{yearly_parquet}')
    WHERE payment_type = 1
      AND trip_distance > 0
      AND trip_distance < 100
      AND fare_amount > 0
      AND tip_amount >= 0
      AND tpep_pickup_datetime  IS NOT NULL
      AND tpep_dropoff_datetime IS NOT NULL
      AND tpep_dropoff_datetime > tpep_pickup_datetime
      AND passenger_count > 0
      AND CAST(CAST(RatecodeID  AS NUMERIC) AS INT) IN (1, 2)
  ),
  zones AS (
    SELECT
      CAST(zone_id AS INT)    AS zone_id,
      CAST(borough AS VARCHAR) AS borough,
      CAST(zone_name AS VARCHAR) AS zone_name,
      CAST(wkt AS VARCHAR)    AS wkt
    FROM read_parquet('{zones_path}')
  ),
  enriched AS (
    SELECT
      s.*,

      -- time features
      EXTRACT(hour  FROM pickup_datetime) AS pickup_hour,
      EXTRACT(isodow FROM pickup_datetime) AS dow,
      EXTRACT(month FROM pickup_datetime) AS month_num,

      CAST(EXTRACT(isodow FROM pickup_datetime) IN (6,7) AS INT) AS weekend,
      CAST( (EXTRACT(hour FROM pickup_datetime) BETWEEN 7 AND 9)
         OR (EXTRACT(hour FROM pickup_datetime) BETWEEN 16 AND 18) AS INT) AS rush_hour,
      CAST( (EXTRACT(hour FROM pickup_datetime) >= 22)
         OR (EXTRACT(hour FROM pickup_datetime) <= 5) AS INT) AS night_trip,

      -- duration / speed
      date_diff('second', s.pickup_datetime, s.dropoff_datetime)/60.0 AS duration_min_raw,
      ROUND(date_diff('second', s.pickup_datetime, s.dropoff_datetime)/60.0, 2) AS duration_min,
      ROUND(s.trip_distance / NULLIF(date_diff('second', s.pickup_datetime, s.dropoff_datetime)/3600.0, 0), 2) AS speed_mph,
      pu.borough   AS pickup_borough,
      doz.borough   AS dropoff_borough,
      pu.zone_name AS pickup_zone_name,
      doz.zone_name AS dropoff_zone_name,

      -- airport-ish flag (JFK/LGA/"airport" in either zone)
      CAST(
        lower(COALESCE(pu.zone_name,'')) LIKE '%airport%'     OR lower(COALESCE(doz.zone_name,'')) LIKE '%airport%'
        OR lower(COALESCE(pu.zone_name,'')) LIKE '%jfk%'      OR lower(COALESCE(doz.zone_name,'')) LIKE '%jfk%'
        OR lower(COALESCE(pu.zone_name,'')) LIKE '%laguardia%' OR lower(COALESCE(doz.zone_name,'')) LIKE '%laguardia%'
      AS INT) AS is_airport_trip,

      -- tip rate
      s.tip_amount / NULLIF(s.fare_amount, 0) AS tip_rate

    FROM src s
    LEFT JOIN zones pu  ON pu.zone_id  = s.pickup_location_id
    LEFT JOIN zones doz ON doz.zone_id = s.dropoff_location_id
    WHERE
      date_diff('minute', s.pickup_datetime, s.dropoff_datetime) BETWEEN 1 AND 300
      AND (s.trip_distance / NULLIF(date_diff('second', s.pickup_datetime, s.dropoff_datetime)/3600.0, 0)) BETWEEN 0 AND 60
  ),
  flags AS (
    SELECT
      e.*,
      CASE
        WHEN {year}=2022 AND e.pickup_datetime >= TIMESTAMP '{CUTOVER_2022}'
          THEN {INITIAL_2022_NEW} ELSE {INITIAL_DEFAULT}
      END AS initial_charge,
      CASE
        WHEN {year}=2022 AND e.pickup_datetime >= TIMESTAMP '{CUTOVER_2022}'
          THEN {UNIT_2022_NEW} ELSE {UNIT_RATE_DEFAULT}
      END AS unit_rate,
      CASE
        WHEN {year}=2022 AND e.pickup_datetime >= TIMESTAMP '{CUTOVER_2022}'
          THEN {JFK_FLAT_2022} ELSE {JFK_FLAT_DEFAULT}
      END AS jfk_flat,
      FLOOR(GREATEST(e.trip_distance / {MI_PER_TICK},
                     (e.duration_min_raw * 60.0) / {SEC_PER_TICK}))::INT AS min_ticks,
      CEIL (GREATEST(e.trip_distance / {MI_PER_TICK},
                     (e.duration_min_raw * 60.0) / {SEC_PER_TICK}))::INT AS max_ticks,
      LEAST(GREATEST(COALESCE(e.extra, 0.0), 0.0), {EXTRA_CAP})
        + LEAST(GREATEST(COALESCE(e.mta_tax, 0.0), 0.0), {MTA_TAX_CAP})
        + LEAST(GREATEST(COALESCE(e.improvement_surcharge, 0.0), 0.0), {IMPROVEMENT_CAP})
        + LEAST(GREATEST(COALESCE(e.congestion_surcharge, 0.0), 0.0), {CONGESTION_CAP})
        + LEAST(GREATEST(COALESCE(e.airport_fee, 0.0), 0.0), {AIRPORT_FEE_CAP}) AS capped_addons
    FROM enriched e
  ),

  bounds AS (
    SELECT
      f.*,
      (initial_charge + unit_rate * min_ticks) AS min_fare_tick,
      (initial_charge + unit_rate * max_ticks) AS max_fare_tick
    FROM flags f
  ),

  jfk_override AS (
    SELECT
      b.*,
      CASE
        WHEN rate_code = 2
         AND (
              (pickup_location_id = 132 AND dropoff_borough = 'Manhattan')
           OR (dropoff_location_id = 132 AND pickup_borough = 'Manhattan')
         )
        THEN b.jfk_flat ELSE NULL
      END AS jfk_fare
    FROM bounds b
  ),

  final_filter AS (
    SELECT
      j.*,
      CASE
        WHEN j.jfk_fare IS NOT NULL THEN j.jfk_fare
        ELSE GREATEST(initial_charge, LEAST(min_fare_tick, max_fare_tick))
      END AS meter_lo_raw,
      CASE
        WHEN j.jfk_fare IS NOT NULL THEN (j.jfk_fare + j.capped_addons)
        ELSE (GREATEST(min_fare_tick, max_fare_tick) + j.capped_addons)
      END AS meter_hi_raw
    FROM jfk_override j
  ),

  trimmed AS (
    SELECT
      *,
      meter_lo_raw * (1.0 - {tolerance_pct}) AS meter_lo,
      meter_hi_raw * (1.0 + {tolerance_pct}) AS meter_hi
    FROM final_filter
    WHERE fare_amount BETWEEN (meter_lo_raw * (1.0 - {tolerance_pct}))
                         AND (meter_hi_raw * (1.0 + {tolerance_pct}))
  )

  SELECT
    pickup_datetime,
    dropoff_datetime,
    TRY_CAST(passenger_count AS TINYINT)   AS passenger_count,
    CAST(rate_code AS TINYINT)           AS rate_code,
    CAST(pickup_location_id AS SMALLINT) AS pickup_location_id,
    CAST(dropoff_location_id AS SMALLINT) AS dropoff_location_id,
    pickup_borough,
    pickup_zone_name,
    dropoff_borough,
    dropoff_zone_name,
    CAST(is_airport_trip AS BOOLEAN)    AS is_airport_trip,
    CAST(pickup_hour AS TINYINT)        AS pickup_hour,
    CAST(dow AS TINYINT)                AS dow,
    CAST(month_num AS TINYINT)          AS month_num,
    CAST(weekend AS BOOLEAN)            AS weekend,
    CAST(rush_hour AS BOOLEAN)          AS rush_hour,
    CAST(night_trip AS BOOLEAN)         AS night_trip,
    CAST(trip_distance AS FLOAT)        AS trip_distance,
    CAST(duration_min AS FLOAT)         AS duration_min,
    CAST(speed_mph AS FLOAT)            AS speed_mph,
    CAST(fare_amount AS DECIMAL(9,2))   AS fare_amount,
    CAST(extra AS DECIMAL(9,2))         AS extra,
    CAST(mta_tax AS DECIMAL(9,2))       AS mta_tax,
    CAST(improvement_surcharge AS DECIMAL(9,2)) AS improvement_surcharge,
    CAST(congestion_surcharge AS DECIMAL(9,2))  AS congestion_surcharge,
    CAST(airport_fee AS DECIMAL(9,2))           AS airport_fee,
    CAST(tolls_amount AS DECIMAL(9,2))  AS tolls_amount,
    CAST(tip_amount AS DECIMAL(9,2))    AS tip_amount,
    CAST(total_amount AS DECIMAL(9,2))  AS total_amount,
    CAST(tip_rate AS FLOAT)             AS tip_rate
  FROM trimmed
  WHERE duration_min > 0 AND duration_min < 300
    AND trip_distance < 100
    AND speed_mph < 60.0
) TO '{out_parquet}'
WITH (FORMAT PARQUET, COMPRESSION 'ZSTD', ROW_GROUP_SIZE {row_group});
"""

def clean_year_with_duckdb(year , row_group=256_000):
    con = duckdb.connect()
    for p in PRAGMAS:
        con.execute(p)

    yearly_parquet = pathlib.Path(YEARLY_DIR, f"yellow_year_{year}.parquet").as_posix()
    out_parquet    = pathlib.Path(OUT_DIR,   f"yellow_clean_{year}.parquet").as_posix()

    sql = SQL_TEMPLATE.format(
        pragmas=";\n".join(PRAGMAS)+";",
        zones_path=ZONES_PATH,
        yearly_parquet=yearly_parquet,
        year=year,
        INITIAL_DEFAULT = INITIAL_DEFAULT,
        UNIT_RATE_DEFAULT = UNIT_RATE_DEFAULT,
        INITIAL_2022_NEW  = INITIAL_2022_NEW,
        UNIT_2022_NEW     = UNIT_2022_NEW,
        MI_PER_TICK       = MI_PER_TICK,
        SEC_PER_TICK      = SEC_PER_TICK,
        JFK_FLAT_DEFAULT  = JFK_FLAT_DEFAULT,
        JFK_FLAT_2022     = JFK_FLAT_2022,
        CUTOVER_2022      = CUTOVER_2022,
        EXTRA_CAP         = EXTRA_CAP,
        MTA_TAX_CAP       = MTA_TAX_CAP,
        IMPROVEMENT_CAP   = IMPROVEMENT_CAP,
        CONGESTION_CAP    = CONGESTION_CAP,
        AIRPORT_FEE_CAP   = AIRPORT_FEE_CAP,
        tolerance_pct=0.20,
        out_parquet=out_parquet,
        row_group=row_group
    )
    con.execute(sql)
    # quick count check
    result = con.execute(f"SELECT count(*) FROM read_parquet('{out_parquet}')").fetchone()
    n = result[0] if result else 0
    print(f"{year}: wrote {out_parquet}  rows={n:,}")
    con.close()

for y in YEARS:
    clean_year_with_duckdb(y, row_group=256_000)
