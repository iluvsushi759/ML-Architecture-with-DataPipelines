**DataPipeline and ML Integration (still in process)**   
  
This repo is a demo of how to perform CI/CD. These are the following scripts that have been created for this Repo:  
  
.github/workflows/snowflake-ci-cd.yml  
ai_agents/agent.py   
ai_agents/commands.py  
ai_agents/rag_setup  
artifacts/metrics.json  > 💡 **Note:** This will be auto generated from your train.py   
artifacts/model.joblib  > 💡 **Note:** This will be auto generated from your train.py or train_optuna.py  
ml/app.py  
ml/evaluate.py  
ml/inference.py  
ml/preprocess.py  
ml/sagemaker_job.py  
ml/train_optuna.py  
ml/train.py  
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

***ML Application***
This will go through different training types.  You will also have different inference options.  
I've also included a RAG wrapped up in an AI agent.

***Look at README2.md for more detailed breakdown.***