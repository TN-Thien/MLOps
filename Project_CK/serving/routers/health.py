from fastapi import APIRouter

router_suc_khoe = APIRouter(tags=["Sức khoẻ"])

@router_suc_khoe.get("/suc-khoe")
@router_suc_khoe.get("/health")
def suc_khoe():
    return {"trang_thai": "ok"}