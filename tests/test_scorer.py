import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARTIFACTS_PRESENT = (Path(__file__).resolve().parent.parent / "models" / "artifacts" / "metadata.json").exists()

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="requires trained model artifacts (run models/train.py first)"
)


@pytest.fixture(scope="module")
def scorer():
    from api.scorer import FraudScorer
    return FraudScorer()


def test_scorer_returns_expected_keys(scorer):
    txn = {
        "cc_num": 9999, "amt": 300.0, "distance_km": 2.0, "age": 30.0,
        "hour_of_day": 12, "day_of_week": 1, "category": "grocery", "unix_time": 1_760_000_000.0,
    }
    result = scorer.score_transaction(txn)
    for key in ("xgb_prob", "seq_prob", "fraud_probability", "is_fraud_predicted", "threshold", "latency_ms"):
        assert key in result
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_scorer_flags_obvious_fraud_pattern(scorer):
    legit = {
        "cc_num": 8888, "amt": 200.0, "distance_km": 1.0, "age": 30.0,
        "hour_of_day": 13, "day_of_week": 2, "category": "grocery", "unix_time": 1_760_000_100.0,
    }
    fraud = {
        "cc_num": 8888, "amt": 95000.0, "distance_km": 2200.0, "age": 30.0,
        "hour_of_day": 2, "day_of_week": 2, "category": "jewellery", "unix_time": 1_760_000_200.0,
    }
    legit_result = scorer.score_transaction(legit)
    fraud_result = scorer.score_transaction(fraud)
    assert fraud_result["fraud_probability"] > legit_result["fraud_probability"]
    assert fraud_result["is_fraud_predicted"] is True


def test_scoring_latency_is_fast(scorer):
    txn = {
        "cc_num": 7777, "amt": 500.0, "distance_km": 3.0, "age": 40.0,
        "hour_of_day": 15, "day_of_week": 3, "category": "fuel", "unix_time": 1_760_000_300.0,
    }
    result = scorer.score_transaction(txn)
    assert result["latency_ms"] < 100  # generous CI-safe bound; typical local latency is ~1-5ms
