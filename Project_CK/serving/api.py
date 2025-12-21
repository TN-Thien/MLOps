from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving.routers.health import router_health
from serving.routers.root import router_root
from serving.routers.predict import router_predict_api, router_predict_ui

tags_metadata = [
    {"name": "health", "description": "Health check endpoints"},
    {"name": "predict", "description": "Prediction endpoints"},
]

app = FastAPI(
    title="IQA API",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.include_router(router_health)
app.include_router(router_root)
app.include_router(router_predict_api)
app.include_router(router_predict_ui)

app.mount("/static", StaticFiles(directory="serving/static"), name="static")