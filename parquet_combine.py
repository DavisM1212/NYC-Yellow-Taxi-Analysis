import os, glob, pathlib
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SRC_ROOT = "./taxi_parquet_cache"     # monthly parquets location
OUT_ROOT = "./taxi_parquets"     # where to write yearly parquets
os.makedirs(OUT_ROOT, exist_ok=True)

YEARS = range(2015, 2023)

KEEP_COLS = [
    "tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance",
    "RatecodeID","PULocationID","DOLocationID","fare_amount","extra","mta_tax",
    "improvement_surcharge","congestion_surcharge","airport_fee",
    "tip_amount","tolls_amount","total_amount","payment_type"
]


TARGET_TYPES = {
    "tpep_pickup_datetime": pa.timestamp("ns"),
    "tpep_dropoff_datetime": pa.timestamp("ns"),
    "passenger_count": pa.float64(),
    "trip_distance": pa.float64(),
    "RatecodeID": pa.int32(),
    "PULocationID": pa.int32(),
    "DOLocationID": pa.int32(),
    "fare_amount": pa.float64(),
    "extra": pa.float64(),
    "mta_tax": pa.float64(),
    "improvement_surcharge": pa.float64(),
    "congestion_surcharge": pa.float64(),
    "airport_fee": pa.float64(),
    "tip_amount": pa.float64(),
    "tolls_amount": pa.float64(),
    "total_amount": pa.float64(),
    "payment_type": pa.int32(),
}

# Streaming settings
BATCH_ROWS = 100_000         # rows per stream batch read
ROW_GROUP_ROWS = 256_000     # rows per parquet row group

def _ensure_schema(schema: pa.schema, keep_cols, target_types):
    fields = []
    for name in keep_cols:
        if name not in schema.names:
            # If a column is absent in some months, create it as nulls later
            fields.append(pa.field(name, target_types[name]))
        else:
            fields.append(pa.field(name, target_types[name]))
    return pa.schema(fields)

def _cast_table_to(table: pa.Table, target_schema: pa.Schema):
    cols = []
    for i, field in enumerate(target_schema):
        name = field.name
        if name in table.column_names:
            col = table[name]
            if not col.type.equals(field.type):
                # cast with safe=False to coerce ints/floats
                col = pc.cast(col, field.type, safe=False)
        else:
            # insert null column if source month is missing this field
            col = pa.nulls(len(table)).cast(field.type)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=target_schema)

def combine_year_stream(year):
    month_glob = os.path.join(SRC_ROOT, str(year), f"yellow_tripdata_{year}-*.parquet")
    files = sorted(glob.glob(month_glob))
    if not files:
        raise FileNotFoundError(f"No Parquet files found for {year} at {month_glob}")

    out_path = pathlib.Path(OUT_ROOT, f"yellow_year_{year}.parquet").as_posix()

    # Build a dataset across all month files
    dataset = ds.dataset(files, format="parquet")

    # Derive a target schema
    target_schema = _ensure_schema(dataset.schema, KEEP_COLS, TARGET_TYPES)

    # Writer with ZSTD compression
    writer = pq.ParquetWriter(out_path, target_schema, compression="zstd", write_statistics=True)

    total_in = 0
    try:
        # Stream month-by-month
        for f in files:
            part_ds = ds.dataset([f], format="parquet")
            # Read batches for this file
            for batch in part_ds.to_batches(columns=KEEP_COLS, batch_size=BATCH_ROWS):
                total_in += batch.num_rows
                tbl = pa.Table.from_batches([batch])
                tbl = _cast_table_to(tbl, target_schema)
                # Write in row-group chunks—split if batch is bigger than ROW_GROUP_ROWS
                start = 0
                n = tbl.num_rows
                while start < n:
                    end = min(start + ROW_GROUP_ROWS, n)
                    writer.write_table(tbl.slice(start, end - start), row_group_size=None)  # already slicing to RG size
                    start = end
    finally:
        writer.close()

    # Verify
    out_rows = pq.ParquetFile(out_path).metadata.num_rows
    print(f"{year}: wrote {out_path}  rows_in≈{total_in:,}  rows_out={out_rows:,}")

for y in YEARS:
    combine_year_stream(y)
