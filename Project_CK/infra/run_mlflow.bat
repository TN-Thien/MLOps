@echo off
setlocal

if not exist mlflow_local\artifacts mkdir mlflow_local\artifacts

set BACKEND=sqlite:///mlflow_local/mlflow.db
set ARTIFACT=./mlflow_local/artifacts

echo Starting MLflow...
mlflow server ^
  --backend-store-uri %BACKEND% ^
  --default-artifact-root %ARTIFACT% ^
  --host 127.0.0.1 ^
  --port 5000