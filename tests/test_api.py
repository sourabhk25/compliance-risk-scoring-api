from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_predict_single():
    payload = {
        "txn_amount": 950.0,
        "txn_count_24h": 18,
        "account_age_days": 12,
        "is_international": 1,
        "device_change_7d": 1,
        "failed_logins_24h": 4
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "risk_score" in data
    assert "risk_label" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] in {"LOW", "MEDIUM", "HIGH"}

def test_predict_batch():
    payload = {
        "items": [
            {
                "txn_amount": 25.5,
                "txn_count_24h": 1,
                "account_age_days": 900,
                "is_international": 0,
                "device_change_7d": 0,
                "failed_logins_24h": 0
            },
            {
                "txn_amount": 420.0,
                "txn_count_24h": 7,
                "account_age_days": 30,
                "is_international": 1,
                "device_change_7d": 1,
                "failed_logins_24h": 2
            }
        ]
    }
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "risk_score" in item
        assert "risk_label" in item
        assert 0.0 <= item["risk_score"] <= 1.0
        assert item["risk_label"] in {"LOW", "MEDIUM", "HIGH"}
