from airflow import DAG
from airflow.operators.bash import BashOperator
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


with DAG(
    dag_id="multimodal_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    tika = BashOperator(
        task_id="tika_extract",
        bash_command=f"""
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate py310
        python {PROJECT_DIR}/tika_extract.py
        """
    )

    spark = BashOperator(
        task_id="spark_fusion",
        cwd=PROJECT_DIR,
        bash_command=f"""
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate py310

        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
        export PYSPARK_PYTHON=$(which python)
        export PYSPARK_DRIVER_PYTHON=$(which python)

        spark-submit {PROJECT_DIR}/spark_fusion.py
        """
    )


    # spark = BashOperator(
    #     task_id="spark_fusion",
    #     cwd=PROJECT_DIR,
    #     bash_command=f"""
    #     source {VENV_DIR}/bin/activate

    #     export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
    #     export SPARK_HOME={VENV_DIR}
    #     export PYSPARK_PYTHON={VENV_DIR}/bin/python
    #     export PYSPARK_DRIVER_PYTHON={VENV_DIR}/bin/python

    #     {VENV_DIR}/bin/spark-class \
    #     org.apache.spark.deploy.SparkSubmit \
    #     spark_fusion.py
    #     """
    # )

     # singa = BashOperator(
    #     task_id="singa_inference",
    #     bash_command="python singa_model.py"
    # )

    tika >> spark
