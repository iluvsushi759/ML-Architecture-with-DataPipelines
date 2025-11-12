import os
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session

def launch_training_job():
    role_arn = os.environ['SAGEMAKER_EXECUTION_ROLE_ARN']
    bucket = os.environ['SAGEMAKER_BUCKET']
    prefix = os.environ.get('SAGEMAKER_PREFIX', 'insurance-claims')

    sess = Session()
    sklearn = SKLearn(
        entry_point='train.py',
        source_dir='ml',
        role=role_arn,
        framework_version='1.2-1',
        instance_type=os.environ.get('SAGEMAKER_INSTANCE_TYPE', 'ml.m5.xlarge'),
        instance_count=int(os.environ.get('SAGEMAKER_INSTANCE_COUNT', '1')),
        output_path=f"{bucket}/{prefix}/output",
        hyperparameters={}
    )
    job_name = os.environ.get('SAGEMAKER_JOB_NAME', sagemaker.utils.name_from_base('claims-xgb'))
    sklearn.fit(job_name=job_name)
    print(f"Launched SageMaker job: {job_name}")

if __name__ == "__main__":
    launch_training_job()
