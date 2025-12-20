import io
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

from serving.api import app

def test_du_doan_tra_diem():
    # Tạo ảnh giả trong bộ nhớ
    img = Image.new("RGB", (384, 512), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # Mock model: trả 0.5 => 50.0 điểm
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.5]])

    with patch("serving.routers.predict.load_model", return_value=mock_model):
        client = TestClient(app)
        r = client.post("/du-doan", files={"file": ("a.jpg", buf, "image/jpeg")})
        assert r.status_code == 200
        data = r.json()
        assert "diem_chat_luong" in data
        assert data["diem_chat_luong"] == 50.0
