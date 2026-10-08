import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACTS_PRESENT = (Path(__file__).resolve().parent.parent / "models" / "artifacts" / "metadata.json").exists()
pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="requires trained model artifacts (run models/train.py first)"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # isolate the test DB from the dev DB
    db_path = tmp_path_factory.mktemp("db") / "test_cardsentry.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c
    os.environ.pop("DATABASE_URL", None)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_endpoint_persists_and_returns_probability(client):
    payload = {
        "cc_num": 123456789, "first": "Test", "last": "User", "merchant": "Big Basket",
        "category": "grocery", "amt": 450.0, "locality": "Adyar", "distance_km": 2.5, "age": 28,
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "fraud_probability" in body
    assert body["merchant"] == "Big Basket"

    recent = client.get("/recent?limit=5").json()
    assert len(recent) >= 1
    assert recent[0]["merchant"] == "Big Basket"


def test_stats_endpoint(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "transaction_count" in body
    assert "fraud_rate" in body


def test_score_rejects_invalid_amount(client):
    payload = {
        "cc_num": 1, "merchant": "X", "category": "grocery", "amt": -5.0,
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 422
