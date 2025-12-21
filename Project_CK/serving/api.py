from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving.routers.health import router_suc_khoe
from serving.routers.root import router_trang_chu
from serving.routers.predict import router_du_doan_api, router_du_doan_ui

app = FastAPI(
    title="IQA – Đánh giá chất lượng hình ảnh (MLOps Local)",
    version="1.0.0",
)

app.include_router(router_suc_khoe)
app.include_router(router_trang_chu)
app.include_router(router_du_doan_api)
app.include_router(router_du_doan_ui)

app.mount("/static", StaticFiles(directory="serving/static"), name="static")