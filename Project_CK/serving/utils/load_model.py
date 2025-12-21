import os
from functools import lru_cache
import torch
import mlflow.pytorch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("TEN_MODEL_IQA", "iqa_efficientnet_b0")
MODEL_ALIAS = os.getenv("ALIAS_IQA", "staging")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device)
    model.eval()
    return model
