# sagemaker_job.py
import os
import time
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session
from sagemaker.s3 import S3Uploader

def upload_artifacts_to_s3(local_dir, bucket, prefix):
    s3_uri = f"s3://{bucket}/{prefix}/input"
    print(f"Uploading {local_dir} -> {s3_uri} ...")
    S3Uploader.upload(local_dir, s3_uri, recursive=True)
    return s3_uri

def launch_training_job(
    role_arn=None,
    bucket=None,
    prefix="insurance-claims",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    job_name=None,
    local_artifacts_dir="artifacts",
    wait_for_completion=True,
    max_run_seconds=None
):
    role_arn = role_arn or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    bucket = bucket or os.environ.get("SAGEMAKER_BUCKET")
    if not role_arn or not bucket:
        raise ValueError("SAGEMAKER_EXECUTION_ROLE_ARN and SAGEMAKER_BUCKET must be set")

    sess = Session()
    sagemaker_session = sagemaker.session.Session(boto_session=sess.boto_session, default_bucket=bucket)
    s3_input_uri = upload_artifacts_to_s3(local_artifacts_dir, bucket, prefix)

    sklearn = SKLearn(
        entry_point="train.py",
        source_dir=".",                # assumes train.py lives in repo root or adjust to 'ml'
        role=role_arn,
        framework_version="1.2-1",
        py_version="py3",
        instance_type=instance_type,
        instance_count=int(instance_count),
        output_path=f"s3://{bucket}/{prefix}/output",
        hyperparameters={}
    )

    job_name = job_name or os.environ.get("SAGEMAKER_JOB_NAME") or sagemaker.utils.name_from_base("claims-xgb")
    print(f"Starting SageMaker SKLearn job: {job_name}")

    sklearn.fit(inputs={"training": s3_input_uri}, job_name=job_name, wait=False)

    if wait_for_completion:
        print("Waiting for training to finish (streaming logs)...")
        training_job_name = sklearn.latest_training_job.name
        sagemaker_session.logs_for_job(training_job_name, wait=True, poll=10)
        desc = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        status = desc["TrainingJobStatus"]
        print(f"Training job {training_job_name} finished with status: {status}")
        return desc
    else:
        return {"TrainingJobName": job_name, "S3Input": s3_input_uri}

if __name__ == "__main__":
    # convenience CLI-style usage using env vars:
    launch_training_job(
        role_arn=os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN"),
        bucket=os.environ.get("SAGEMAKER_BUCKET"),
        prefix=os.environ.get("SAGEMAKER_PREFIX", "insurance-claims"),
        instance_type=os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.m5.xlarge"),
        instance_count=int(os.environ.get("SAGEMAKER_INSTANCE_COUNT", "1")),
        job_name=os.environ.get("SAGEMAKER_JOB_NAME"),
        local_artifacts_dir=os.environ.get("LOCAL_ARTIFACTS_DIR", "artifacts"),
        wait_for_completion=True
    )