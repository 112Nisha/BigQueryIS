# BigQueryIS

## Set-up
```
pip install tika
pip install pyspark
pip install apache-airflow
```

**Environment Variables**
```
export AIRFLOW_HOME=<path/to/dir/airflow> # note the airflow dir
```

## Running Airflow
```
airflow db reset -y
airflow standalone
```
Now for the username and pwd, check the airflow directory on home (or wherever you installed airflow) and run
`/airflow/simple_auth_manager_passwords.json.generated`




## Dataset
[Yelp Dataset](https://business.yelp.com/data/resources/open-dataset/)