from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from serving.config import templates

router_trang_chu = APIRouter(tags=["Giao diện"])

@router_trang_chu.get("/", response_class=HTMLResponse)
def trang_chu(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
