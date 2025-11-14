**DataPipeline and ML Integration (still in process)**   

This repo is a demo of how to perform CI/CD. There are the following scripts that have been created within this Repo:  
  
.github/workflows/ml-ci-cd.yml  
.github/workflows/snowflake-ci-cd.yml  
agents/data_quality_agent.py  
agents/orchestration.py  
artifacts/metrics.json  > 💡 **Note:** This will be auto generated from your train.py   
artifacts/model.joblib  > 💡 **Note:** This will be auto generated from your train.py or train_optuna.py  
ml/app.py  
ml/evaluate.py  
ml/inference.py  
ml/model_registry.py  
ml/preprocess.py  
ml/sagemaker_job.py  
ml/train_optuna.py  
ml/train.py  
rag/embed.py  
rag/llm_chain.py  
rag/vector_store.py  
scripts/run_sql.py  
sql/00_create_schemas.sql  
sql/01_create_tables.sql  
sql/02_load_raw.sql  
sql/03_transform.sqlsql  
sql/04_merge_into_presentation.sql  
sql/05_data_quality_checks.sql  
  
***Data Creation***  
Now in regards to data, I've just generated sythetic data and exported the .csv files in the repo root.  Feel free to change them or add more.   
  
***Activation***  
YML script is placed in the .github/workflow directory. This is where we will control our workflow .sql. This will also create your staging and production schema and tables in Snowflake.    
1. Flow of process:  dump to AWS s3 source data-> Snowflake -> CI/CD -> Dev will then use ML to train data from Snowflake -> Inference for prediction.  
2. My model training with the train.py is based on XGBoost/XGBRegressor.  From there you can perform the inference (inference.py)  
3. Now I've also decided to perform HyperParameter tuning to attempt to make this improve itself via train_optuna.py.  As you can see in the file, it will run 50 trials By the way, I set it for CPU but I suggest you do GPU if you can:   
   
params.update({  
    "tree_method": "gpu_hist",  
    "predictor": "gpu_predictor"  
})  
  
4. We can also dashboard it via streamlit using app.py and will give you a "webpage"  which shows your results interactively, Root Mean Square Errors, Prediction vs Actuals Validation vs Training.    
  
***Linting Option***  
I'm using SQLFluff and unfortunately, it's VERY strict, so I created this config file to make exceptions.  
  
***Script Sequence***  
  
0️⃣ 00_create_schemas.sql -- yea I know. Step 0?  Well I was creating scripts and just decided to use that since I was already creating an 00 script. LOL.  This will create or replace your schemas  
  
1️⃣ 01_create_tables.sql -- Creates the core database tables in Snowflake, defining schema structures, data types, and constraints. This establishes the foundational data model for all subsequent operations.  
  
2️⃣ 02_load_raw.sql -- Truncates RAW tables (to ensure no data), Loads initial datasets into the newly created tables using Snowflake’s Storage Integration.  Make sure you create your storage integration before running this script.  Your Snowflake Integration is going to be up to you since you may end up with AWS, Azure or GCP.  You storage ARN will also  be different than mine.  This step copies raw data files from cloud storage (e.g., AWS S3) into Snowflake RAW tables.  I've placed other sanity checks and in this script that won't be needed but you can leave that in, in case you want to run these script individually.  
  
3️⃣ 03_transform.sql -- Performs transformations and joins across the loaded tables. Typically this creates views or derived tables that represent curated, business-ready datasets.  
  
4️⃣ 04_merge_into_presentation_sql -- This represents the curated data  
  
5️⃣ 05_data_quality_checks.sql -- Runs basic data validation checks (such as row counts, null checks, and data type verification) to confirm that the data was successfully loaded and transformed.  
  
***Process and Execution***  
Once an object or code is updated or deleted, the continuous integration will perform it's checks and linting then your deployment will begin. If all passes, schedule your change control, go through your PULL request to merge with your master/prod branch.  
  
***Why This Approach***  
  
This modular, step-by-step pipeline offers several benefits:  
  
 - Clear Separation of Concerns: Each script has a single purpose—creation, loading, transformation, or validation—making debugging and maintenance easier.  
 - Reproducible CI/CD: The workflow can be re-run from scratch at any time with consistent results.  
 - Scalable Design: As new tables or transformations are added, new scripts can be appended without impacting existing steps.  
 - Transparency and Auditability: Each stage is visible in GitHub and traceable in CI/CD logs, ensuring full visibility into what’s deployed and why.  
 - Data Quality Assurance: The final validation step ensures that downstream consumers only interact with verified data.  
  
***Application***   
  
This demo can be also used in other applications such as:  
  
 - inserting or changing code in your terminal and pushing this to have immediate testing   
 - The ML will adjust accordingly to the changes your make to the data and vice versa along with the previous mentioned.  
 - ROLLBACK option - if we make a change we can easily re-apply our former state. As long as the older commit that is pointing to still has it's files within S3 and hasn't changed, we can perform this. This is a code change tracker where we can revert structure and possibly data.  

Pull Request to main:  
 - Team reviews the change (code review, lint checks, CI results).  
 - Ensures the change is intentional, correct, and compliant.