# export_curated_to_s3.py
import snowflake.connector
import boto3

# -----------------------------
# 1️⃣ Configuration
# -----------------------------
SNOWFLAKE_CONFIG = {
    "user": "iluvsushi",
    "password": "Kamronsui1!!!!",
    "account": "nkoyqld-toc82072",
    "warehouse": "COMPUTE_WH",
    "database": "INSURANCE_DB",
    "schema": "CURATED"
}

S3_BUCKET = "sagemaker-insurance-curated-from-snowflake"
TABLES_TO_EXPORT = ["CLAIMS_CLEAN", "PATIENTS", "TRIALS"]  # from your doc
FILE_FORMAT = "PARQUET"  # fast, SageMaker-friendly
COMPRESSION = "SNAPPY"

# -----------------------------
# 2️⃣ Create S3 bucket if missing
# -----------------------------
s3 = boto3.client("s3")
existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

if S3_BUCKET not in existing_buckets:
    s3.create_bucket(Bucket=S3_BUCKET)
    print(f"Created bucket: {S3_BUCKET}")
else:
    print(f"Bucket exists: {S3_BUCKET}")

# -----------------------------
# 3️⃣ Connect to Snowflake
# -----------------------------
conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
cs = conn.cursor()

# -----------------------------
# 4️⃣ Export each curated table to S3
# -----------------------------
for table in TABLES_TO_EXPORT:
    s3_path = f"s3://{S3_BUCKET}/{table.lower()}/"
    sql = f"""
    COPY INTO '{s3_path}'
    FROM {SNOWFLAKE_CONFIG['schema']}.{table}
    FILE_FORMAT = (TYPE = {FILE_FORMAT} COMPRESSION = {COMPRESSION} HEADER = TRUE)
    OVERWRITE = TRUE;
    """
    print(f"Exporting {table} -> {s3_path} ...")
    cs.execute(sql)
    print(f"{table} export complete!")

cs.close()
conn.close()
print("All curated tables exported to S3.")
