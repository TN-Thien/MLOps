import io
import torch
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from torchvision import transforms

from serving.config import templates
from serving.utils.load_model import load_model, device

router_du_doan_api = APIRouter(tags=["Dự đoán"])
router_du_doan_ui = APIRouter(tags=["Giao diện"])

_transform = transforms.Compose(
    [
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

@router_du_doan_ui.get("/du-doan", response_class=HTMLResponse)
@router_du_doan_ui.get("/predict-ui", response_class=HTMLResponse)
def giao_dien_du_doan(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@router_du_doan_api.post("/du-doan")
@router_du_doan_api.post("/predict")
async def du_doan(file: UploadFile = File(...)):
    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="File không phải hình ảnh.")

    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh. Hãy thử file khác.")

    x = _transform(img).unsqueeze(0).to(device)

    model = load_model()
    with torch.no_grad():
        pred = model(x).squeeze().item()

    diem = float(pred * 100.0)
    return {"diem_chat_luong": round(diem, 2)}