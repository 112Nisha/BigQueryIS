from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime
import os

# You can export this variable to make it more generic
# run these commands on your system for airflow:
# go into a venv, and run 
# pkill -f airflow
# export AIRFLOW_HOME=/home/dell/BigQueryIS/airflow 
# then do airflow db reset and press y
PROJECT_DIR = "/home/dell/BigQueryIS"

with DAG(
    dag_id="multimodal_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    tika = BashOperator(
    task_id="tika_extract",
)

    spark = BashOperator(
        task_id="spark_fusion",
        bash_command=f"spark-submit {PROJECT_DIR}/spark_fusion.py"
    )

     # singa = BashOperator(
    #     task_id="singa_inference",
    #     bash_command="python singa_model.py"
    # )

    tika >> spark
