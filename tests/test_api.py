import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads(
    (ROOT / "examples" / "cancion.json").read_text(encoding="utf-8")
)
def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "Pipeline"
    assert body["features"]
    assert "accuracy" in body["metrics"]
    assert body["model_version"]
def test_predict_valid_song():
    with TestClient(app) as client:
        response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["EsExito"], bool)
    assert 0 <= body["ProbabilidadExito"] <= 1
    assert body["model_version"]
    assert body["timestamp_utc"]
def test_predict_batch():
    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json=[PAYLOAD, PAYLOAD],
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all("EsExito" in result for result in body)
    assert all("ProbabilidadExito" in result for result in body)
def test_predict_invalid_energy():
    invalid_payload = {**PAYLOAD, "energy": 3}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
