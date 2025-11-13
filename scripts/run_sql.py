import os
import snowflake.connector

# Connect to Snowflake using environment variables
conn = snowflake.connector.connect(
    user=os.environ['SNOWFLAKE_USER'],
    password=os.environ['SNOWFLAKE_PASSWORD'],
    account=os.environ['SNOWFLAKE_ACCOUNT'],
    warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
    database=os.environ['SNOWFLAKE_DATABASE'],
    schema=os.environ['SNOWFLAKE_SCHEMA'],
    role=os.environ.get('SNOWFLAKE_ROLE')
)

cur = conn.cursor()
sql_dir = os.environ['SQL_DIR']
target_schema = os.environ['TARGET_SCHEMA']
# This is a fallback default if the environment variable isn’t set. I'mm doing this in case someone runs this script locally and not via .yml

# Ordered execution of SQL files
ordered_files = [
    "00_create_schemas.sql",
    "01_create_tables.sql",
    "02_load_raw.sql",
    "03_transform.sql",
    "04_merge_into_presentation.sql",
    "05_data_quality_checks.sql"
]

for file in ordered_files:
    path = f"{sql_dir}/{file}"
    if os.path.isfile(path):
        print(f"🚀 Running {file} in {target_schema}")
        with open(path, 'r') as f:
            sql = f.read().replace('${TARGET_SCHEMA}', target_schema)
            # Split on semicolons, skip empty statements
            for stmt in sql.split(';'):
                stmt = stmt.strip()
                if stmt:  # only execute non-empty statements
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"⚠️ Error executing statement in {file}: {e}")
                        raise
    else:
        print(f"⚠️ Skipping {file} (not found)")

cur.close()
conn.close()