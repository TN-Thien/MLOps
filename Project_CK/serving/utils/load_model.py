from __future__ import annotations

import os
from functools import lru_cache

import mlflow
import torch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
TEN_MODEL = os.getenv("TEN_MODEL_IQA", "iqa_viet_hoa")
ALIAS = os.getenv("ALIAS_IQA", "thu_nghiem")

device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """
    Load model từ MLflow Model Registry theo alias (VD: thu_nghiem / san_xuat).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{TEN_MODEL}@{ALIAS}"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)
    model.eval()
    return model
