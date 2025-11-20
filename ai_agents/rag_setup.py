import os
import pandas as pd
from snowflake.connector import connect
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------
# Snowflake config
# -----------------------------
SNOWFLAKE_CONFIG = {
    "user": ",input username",
    "password": "input password",
    "account": "input account",
    "warehouse": "input warehouse",
    "database": "input database",
    "schema": "PRESENTATION"
}

# -----------------------------
# Load tables and dataframes
# -----------------------------
def get_tables():
    print("Fetching table list from Snowflake...")
    conn = connect(**SNOWFLAKE_CONFIG)
    cs = conn.cursor()
    cs.execute("SHOW TABLES IN SCHEMA PRESENTATION")
    rows = cs.fetchall()
    columns = [desc[0] for desc in cs.description]
    cs.close()
    conn.close()
    tables_df = pd.DataFrame(rows, columns=columns)
    table_list = tables_df['name'].tolist()
    print(f"Tables found: {table_list}")
    return table_list

def load_table(table_name, limit=None):
    print(f"Loading table: {table_name}...")
    conn = connect(**SNOWFLAKE_CONFIG)
    cs = conn.cursor()
    query = f"SELECT * FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"
    cs.execute(query)
    rows = cs.fetchall()
    columns = [desc[0] for desc in cs.description]
    df = pd.DataFrame(rows, columns=columns)
    cs.close()
    conn.close()
    print(f"{table_name} loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def df_to_documents(df, table_name):
    docs = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(f"[{table_name}] {row_text}")
    return docs

# -----------------------------
# Load scripts as documents
# -----------------------------
def load_scripts_as_documents(folder_path):
    print(f"Loading Python scripts from: {folder_path}...")
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append(f"[Script: {file}]\n{content}")
    print(f"Total scripts loaded: {len(docs)}")
    return docs

# -----------------------------
# Create RAG agent
# -----------------------------
def create_rag(project_folder=None):
    print("Creating RAG agent...")

    # Load all tables
    tables = get_tables()
    dataframes = {table: load_table(table, limit=1000) for table in tables}

    # Convert tables to documents
    print("Converting tables to documents...")
    documents = []
    for table, df in dataframes.items():
        documents.append(f"Table name: {table}")
        documents += df_to_documents(df, table)
    print(f"Table documents created: {len(documents)}")

    # Load project scripts as documents
    if project_folder:
        script_docs = load_scripts_as_documents(project_folder)
        documents += script_docs
    print(f"Total documents including scripts: {len(documents)}")

    # Split documents into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunked_docs = []
    for doc in documents:
        chunked_docs += text_splitter.split_text(doc)
    print(f"Total chunks created: {len(chunked_docs)}")

    # Embeddings
    print("Creating embeddings...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunked_docs, embeddings_model)
    print("Embeddings created.")

    # LLM setup (Falcon 3-1B)
    print("Loading Falcon 3-1B model...")
    model_name = "tiiuae/Falcon3-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512  # prevent overflow
    )
    llm_model = HuggingFacePipeline(pipeline=pipe)
    print("Model loaded.")

    # RetrievalQA chain
    print("Creating RetrievalQA chain...")
    qa = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    print("RAG agent ready!")
    return qa