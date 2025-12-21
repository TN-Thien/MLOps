@echo off
setlocal
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
set TEN_MODEL_IQA=iqa_efficientnet_b0
set ALIAS_IQA=staging
uvicorn serving.api:app --host 127.0.0.1 --port 8000 --reload