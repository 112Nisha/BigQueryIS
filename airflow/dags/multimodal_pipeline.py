from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime
import os

# You can export this variable to make it more generic
# run these commands on your system for airflow:
# run python3 -m venv venv to get a new venv and then run source ./venv/bin/activate
# pip install apache-airflow
# export AIRFLOW_HOME=/home/dell/BigQueryIS/airflow 
# then do airflow db reset and press y

# run `airflow db reset` after you change multimodel_pipeline.py to see your changes in airflow

DAG_FILE = os.path.abspath(__file__)
DAGS_DIR = os.path.dirname(DAG_FILE)
AIRFLOW_DIR = os.path.dirname(DAGS_DIR)
PROJECT_DIR = os.path.dirname(AIRFLOW_DIR)
VENV_DIR = "cuda"

with DAG(
    dag_id="multimodal_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    tika = BashOperator(
        task_id="tika_extract",
        bash_command=f"""
        source {VENV_DIR}/bin/activate
        python {PROJECT_DIR}/tika_extract.py
        """
    )

    spark = BashOperator(
        task_id="spark_fusion",
        bash_command="""
        set -ux
        echo "PWD=$(pwd)"
        echo "USER=$(whoami)"

        echo "JAVA:"
        java -version || echo "JAVA NOT FOUND"

        echo "SPARK:"
        which spark-submit || echo "spark-submit NOT FOUND"
        ls -l /opt/spark/bin || echo "/opt/spark/bin NOT FOUND"

        echo "PYTHON:"
        source cuda/bin/activate
        which python
        python -V
        """
)

     # singa = BashOperator(
    #     task_id="singa_inference",
    #     bash_command="python singa_model.py"
    # )

    tika >> spark
