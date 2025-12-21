import os
from functools import lru_cache
import mlflow
import torch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEN_MODEL = os.getenv("TEN_MODEL_IQA", "iqa_efficientnet_b0")
ALIAS = os.getenv("ALIAS_IQA", "staging")

device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{TEN_MODEL}@{ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()
    return model
