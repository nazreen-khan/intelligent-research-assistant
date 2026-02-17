from fastapi.testclient import TestClient
from ira.app.main import app

def test_health_has_request_id_and_header():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "request_id" in body
    assert "X-Request-ID" in r.headers
