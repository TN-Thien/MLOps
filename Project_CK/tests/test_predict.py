import io
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

from serving.api import app

def test_predict_returns_quality():
    img = Image.new("RGB", (384, 384), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.5]])

    with patch("serving.routers.predict.load_model", return_value=mock_model):
        client = TestClient(app)
        r = client.post("/predict", files={"file": ("a.jpg", buf, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert "quality" in data
        assert data["quality"] == 50.0

def test_predict_rejects_non_image():
    client = TestClient(app)
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400