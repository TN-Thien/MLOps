import io
import torch
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from torchvision import transforms

from serving.config import templates
from serving.utils.load_model import load_model, device

router_predict_api = APIRouter(tags=["predict"])
router_predict_ui = APIRouter()

transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@router_predict_ui.get("/predict-ui", response_class=HTMLResponse, include_in_schema=False)
def predict_ui(request: Request):
    return templates.TemplateResponse(request, "index.html")

@router_predict_ui.get("/predict", response_class=HTMLResponse, include_in_schema=False)
def predict_alias(request: Request):
    return templates.TemplateResponse(request, "predict.html")

@router_predict_api.post("/predict", summary="Predict image quality")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    model = load_model()

    with torch.no_grad():
        pred = model(x).squeeze().item()

    return {"quality": round(pred * 100.0, 2)}
