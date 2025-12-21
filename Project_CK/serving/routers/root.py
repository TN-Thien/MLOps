from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from serving.config import templates

router_root = APIRouter()

@router_root.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
