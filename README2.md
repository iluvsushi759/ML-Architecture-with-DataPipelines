**Process Flow**  
    
1. I would first advise you create a virtual environment for your python terminal and install your requirements (pip install -r requirements.txt)  
2. The rag_setup.py credentials were hard-carded for snowflake originally.  I've changed this to use environment variable instead. So what you will need to do is run this in your Linux Terminal:  
  
export SNOWFLAKE_ACCOUNT='snowflake account' 
export SNOWFLAKE_USER='snowflake user'  
export SNOWFLAKE_PASSWORD='snowflake password'  
export SNOWFLAKE_DATABASE='snowflake database'  
export SNOWFLAKE_WAREHOUSE='snowflake warehouse'  
export SNOWFLAKE_SCHEMA='snowflake schema'  
  
For windows terminal just change your 'export' to 'set'
  
3. Go to your terminal and navigate to your repo and run:  python scripts/run_sql.py  
4. When the scripts have completed, run your training and cleaning.  The train.py will call a function with the preprocess.py to clean and further curating the data.  You can run this next:  python ml/train.py  
5. You should receive a prompt that training is completed.  For just fast and quick answers, you can do a manual coded inference from running:  python ml/inference.py  
6. If a dashboard with interaction is needed then you can run:  streamlit run ml/app.py  
Feel free to get rid of the training charts but I've included them to see how well the model was doing.  I always like to see how well it's doing when making predictions.
7. If you don't feel happy with the RMSE, I've also included an HPO script, train_optuna.py.  You can run:  python/train_optuna.py  
You should see an improvement with our RMSE.  I've tuned it to 50 trials.  Please feel free to to tweeked this and rerun your streamlit app.py file.  
8. I've provided a sagemaker.py script to run on AWS Sagemaker Jupyter.  This would allow the same thing to run your train.py but with the ability to always tune your Instance size and having better library and framework compatibility.  
  
**RAG and AI Agent Flow**  
1. If you desire to have an agent wrapped around the above process, run your agent.py file via typing in terminal: python -m ai_agents.agent
2. Ask Away!  
  
**Desciption of each file**  
***Script Sequence***  
  
0️⃣ 00_create_schemas.sql -- yea I know. Step 0?  Well I was creating scripts and just decided to use that since I was already creating an 00 script. LOL.  This will create or replace your schemas  
  
1️⃣ 01_create_tables.sql -- Creates the core database tables in Snowflake, defining schema structures, data types, and constraints. This establishes the foundational data model for all subsequent operations.  
  
2️⃣ 02_load_raw.sql -- Truncates RAW tables (to ensure no data), Loads initial datasets into the newly created tables using Snowflake’s Storage Integration.  Make sure you create your storage integration before running this script.  Your Snowflake Integration is going to be up to you since you may end up with AWS, Azure or GCP.  You storage ARN will also  be different than mine.  This step copies raw data files from cloud storage (e.g., AWS S3) into Snowflake RAW tables.  I've placed other sanity checks and in this script that won't be needed but you can leave that in, in case you want to run these script individually.  
  
3️⃣ 03_transform.sql -- Performs transformations and joins across the loaded tables. Typically this creates views or derived tables that represent curated, business-ready datasets.  
  
4️⃣ 04_merge_into_presentation_sql -- This represents the curated data  
  
5️⃣ 05_data_quality_checks.sql -- Runs basic data validation checks (such as row counts, null checks, and data type verification) to confirm that the data was successfully loaded and transformed.  
  
**RAG and AI Agent Description**  
  
1️⃣ agent.py  
Purpose: The main entry point for interacting with your RAG agent.  
  
What it does:  
- Imports create_rag() from rag_setup.py.  
- Initializes the RAG agent.  
- Accepts user queries (via CLI or some input loop).  
- Sends queries to the RetrievalQA chain (qa.invoke()) and returns the answer.  
  
***Key note:*** Doesn’t know about your tables/data itself—it relies on rag_setup.py to build the knowledge base.  
  
2️⃣ rag_setup.py  
Purpose: The heart of the RAG agent setup.  
  
What it does:  
- Connects to Snowflake and retrieves table metadata and rows.  
- Converts tables → Pandas DataFrames → documents → chunked text.  
- Creates embeddings for each chunk using HuggingFace embeddings.  
- Loads the LLM (Falcon 3-1B) and wraps it in a LangChain pipeline.  
- Builds the RetrievalQA chain with the vector store as the retriever.  
- Returns a fully initialized qa object that agent.py can use.  
  
***Key note:*** This script is what “knows” about your data and transforms it into something the LLM can query.  
  
3️⃣ run_sql.py  
Purpose: Utility script for extracting raw data from Snowflake.  
  
What it does:  
- Runs SQL queries on Snowflake tables.  
- Can be used to validate data, check schemas, or prepare CSVs.  
  
***Key note:*** Mostly for testing or generating data snapshots; not directly part of the RAG pipeline but provides the raw data.    
  
4️⃣ train.py  
Purpose: Basic ML training script.  
  
What it does:  
- Trains a model (usually on structured data from Snowflake or CSV exports).  
- Saves the trained model artifacts for later inference.  
  
***Key note:*** Deterministic training—uses predefined hyperparameters.  
  
5️⃣ train_optuna.py  
Purpose: ML training script with hyperparameter optimization.  
  
What it does:  
- Uses Optuna to automatically tune hyperparameters for the model.  
- Evaluates multiple trials to find the best-performing configuration.  
- Saves the optimized model artifacts for later use.  
  
***Key note:*** More advanced than train.py—used when you want the model to be “better than default” automatically.  
  
6️⃣  app.py  
Purpose: Optional front-end / dashboard.  
  
What it does:  
- Exposes your RAG agent through a simple UI or web API.  
- Lets you type queries and see responses in a user-friendly way.  
  
***Key note:*** Not strictly needed for local CLI usage but great for demos or portfolio.  
  
7️⃣ inference.py  
Purpose: Serves as the ML inference engine for predictions.  
  
What it does:  
Loads the trained model and feature metadata from the saved artifacts (model.joblib).  
Provides a predict() function that accepts input data as a dictionary (or JSON string).  
Formats the input to match the model’s expected feature order.  
Runs the model prediction and returns the numeric output.  
    
***Key note:***  
This script is required whenever you want to make predictions from agent.py or app.py.  
It isolates prediction logic from training, so your model can be reused without retraining.  
Handles missing features by defaulting them to 0 and parses JSON input if needed.  

