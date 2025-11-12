import os
import snowflake.connector
import sqlparse

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

# Base files for both CI and CD
base_files = [
    "00_create_schemas.sql",
    "01_create_tables.sql",
    "02_load_raw.sql",
    "03_transform.sql"
]

# Merge runs ONLY when deploying to PRESENTATION (CD job on main)
files = list(base_files)
if target_schema.upper() == "PRESENTATION":
    files.append("04_merge_into_presentation.sql")
    files.append("05_data_quality_checks.sql")

for file in files:
    path = f"{sql_dir}/{file}"
    if os.path.isfile(path):
        print(f"🚀 Running {file} (TARGET_SCHEMA={target_schema})")
        with open(path, 'r') as f:
            sql = f.read().replace('${TARGET_SCHEMA}', target_schema)
            statements = sqlparse.split(sql)
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"❌ Error in {file}: {e}\n Statement: {stmt[:200]}...")
                        raise
    else:
        print(f"⚠️ Skipping {file} (not found)")

cur.close()
conn.close()
