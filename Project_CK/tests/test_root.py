from fastapi.testclient import TestClient
from serving.api import app

def test_root_ok():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()