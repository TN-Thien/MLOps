from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Load model bundle
bundle = joblib.load("diabetes_model.joblib")
model = bundle["model"]
feature_cols = bundle["feature_cols"]
target_col = bundle["target_col"]

# FastAPI app
app = FastAPI(
    title="Diabetes Predictor API",
    description=(
        f"API dự đoán {target_col} từ bộ dữ liệu diabetes. "
        "Mô hình: Logistic Regression trong Pipeline (Imputer + StandardScaler + OneHotEncoder)."
    ),
    version="1.0.0",
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Pydantic model for /predict
class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
    Race: Optional[str] = None

# API endpoint
@app.get("/")
def read_root():
    return {
        "message": f"Mô hình dự đoán {target_col} (diabetes) với FastAPI",
        "feature_cols": feature_cols,
        "note": "POST /predict với JSON đúng các cột trên. Tham khảo /docs hoặc /ui để thử nhanh.",
    }

@app.post("/predict")
def predict(item: InputData):
    payload = item.dict()
    for c in feature_cols:
        payload.setdefault(c, None)
    X = pd.DataFrame([payload], columns=feature_cols)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])
    else:
        proba = float(model.decision_function(X)[0])
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability_of_1": proba, "target_col": target_col}

# HTML UI endpoint
@app.get("/ui", response_class=HTMLResponse)
def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
