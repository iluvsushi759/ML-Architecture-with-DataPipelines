import os
import pandas as pd
import snowflake.connector
from sentence_transformers import SentenceTransformer
from vector_store import init_store, USE_PINECONE

def fetch_rows():
    conn = snowflake.connector.connect(
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema='PRESENTATION',
        role=os.environ.get('SNOWFLAKE_ROLE', None)
    )
    cur = conn.cursor()
    cur.execute("""
      SELECT c.CLAIM_ID, cu.CUSTOMER_ID, h.HOSPITAL_ID, c.CLAIM_TYPE, c.STATUS, c.CLAIM_AMOUNT
      FROM PRESENTATION.CLAIMS c
      LEFT JOIN PRESENTATION.CUSTOMERS cu ON c.CUSTOMER_ID = cu.CUSTOMER_ID
      LEFT JOIN PRESENTATION.HOSPITALS h ON c.HOSPITAL_ID = h.HOSPITAL_ID
    """)
    cols = [c[0] for c in cur.description]
    rows = cur.fetchall()
    cur.close(); conn.close()
    return pd.DataFrame(rows, columns=cols)

def embed_texts(texts):
    if os.environ.get("USE_OPENAI", "false").lower() == "true":
        import requests, json
        api_key = os.environ['OPENAI_API_KEY']
        model = os.environ.get('EMBED_MODEL', 'text-embedding-3-large')
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"input": texts, "model": model}
        r = requests.post("https://api.openai.com/v1/embeddings",
                          headers=headers, data=json.dumps(data))
        r.raise_for_status()
        return [item['embedding'] for item in r.json()['data']]
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(texts, convert_to_numpy=True).tolist()

def upsert_claims():
    df = fetch_rows()
    df['text'] = df.apply(lambda r: f"Claim {r['CLAIM_ID']} customer {r['CUSTOMER_ID']} "
                                    f"hospital {r['HOSPITAL_ID']} type {r['CLAIM_TYPE']} "
                                    f"status {r['STATUS']} amount {r['CLAIM_AMOUNT']}", axis=1)
    texts = df['text'].tolist()
    embeddings = embed_texts(texts)
    store = init_store(dim=len(embeddings[0]))

    if USE_PINECONE:
        vectors = [(str(df['CLAIM_ID'].iloc[i]), embeddings[i], {"table": "CLAIMS"})
                   for i in range(len(df))]
        store.upsert(vectors=vectors)
    else:
        import numpy as np
        store.add(np.array(embeddings))
    print(f"Upserted {len(df)} claims into vector store.")

if __name__ == "__main__":
    upsert_claims()
