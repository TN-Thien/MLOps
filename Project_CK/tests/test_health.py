from fastapi.testclient import TestClient
from serving.api import app

def test_suc_khoe_ok():
    client = TestClient(app)
    r = client.get("/suc-khoe")
    assert r.status_code == 200
    assert r.json()["trang_thai"] == "ok"