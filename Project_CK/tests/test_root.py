from fastapi.testclient import TestClient
from serving.api import app

def test_trang_chu_ok():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Đánh giá chất lượng" in r.text