
import os
import pandas as pd
import snowflake.connector

def fetch_dataframe(sql: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ.get('SNOWFLAKE_SCHEMA', 'PRESENTATION'),
        role=os.environ.get('SNOWFLAKE_ROLE', None)
    )
    cur = conn.cursor()
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    cur.close(); conn.close()
    return df

def build_training_set():
    sql = """
    SELECT
      c.CLAIM_ID,
      cu.AGE,
      CASE WHEN cu.GENDER = 'Male' THEN 1 WHEN cu.GENDER = 'Female' THEN 0 ELSE NULL END AS GENDER_BIN,
      cu.PLAN_TYPE,
      h.BASE_RATING AS HOSPITAL_RATING,
      c.CLAIM_TYPE,
      c.STATUS,
      c.CLAIM_AMOUNT
    FROM PRESENTATION.CLAIMS c
    LEFT JOIN PRESENTATION.CUSTOMERS cu ON c.CUSTOMER_ID = cu.CUSTOMER_ID
    LEFT JOIN PRESENTATION.HOSPITALS h ON c.HOSPITAL_ID = h.HOSPITAL_ID
    WHERE c.CLAIM_AMOUNT IS NOT NULL
    """
    df = fetch_dataframe(sql)

    # ✅ Ensure GENDER_BIN is numeric
    df['GENDER_BIN'] = pd.to_numeric(df['GENDER_BIN'], errors='coerce')

    df = df.dropna(subset=['CLAIM_AMOUNT'])
    df['PLAN_TYPE'] = df['PLAN_TYPE'].fillna('Unknown')
    df['CLAIM_TYPE'] = df['CLAIM_TYPE'].fillna('Unknown')
    df['STATUS'] = df['STATUS'].fillna('Unknown')

    cat_cols = ['PLAN_TYPE', 'CLAIM_TYPE', 'STATUS']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    target = 'CLAIM_AMOUNT'
    feature_cols = [c for c in df.columns if c not in ['CLAIM_ID', target]]
    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols