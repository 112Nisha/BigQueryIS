CREATE DATABASE IF NOT EXISTS multimodal;

CREATE TABLE IF NOT EXISTS multimodal.results (
    customer_id INT,
    prediction DOUBLE
)
STORED AS ORC;
