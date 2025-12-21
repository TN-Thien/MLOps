from fastapi import APIRouter
from serving.utils.load_model import load_model

router_health = APIRouter(tags=["health"])

@router_health.get("/health", summary="Health check")
def health():
    return {"status": "ok"}

@router_health.post("/reload-model", include_in_schema=False)
def reload_model():
    load_model.cache_clear()
    return {"status": "reloaded"}
