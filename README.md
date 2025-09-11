## End to End Data Science project

https://dagshub.com/krrishk/mlproject.mlflow
import dagshub
dagshub.init(repo_owner='krrishk', repo_name='mlproject', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

  MLFLOW_TRACKING_URI=https://dagshub.com/krrishk/mlproject.mlflow
  MLFLOW_TRACKING_USERNAME=krrishk
  MLFLOW_TRACKING_PASSWORD=
  python script.py
