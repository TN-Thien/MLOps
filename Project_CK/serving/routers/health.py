from fastapi import APIRouter

router_health = APIRouter(tags=["health"])

@router_health.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
