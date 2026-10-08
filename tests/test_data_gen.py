import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_gen"))

from data_gen.generate import gen_customers, gen_legit_transactions, gen_fraud_incidents, enrich, haversine_km
from datetime import datetime, timedelta


def test_gen_customers_shape_and_ranges():
    customers = gen_customers(50)
    assert len(customers) == 50
    assert customers["cc_num"].nunique() == 50
    assert customers["lat"].between(10, 15).all()  # roughly Tamil Nadu latitude band
    assert customers["long"].between(78, 82).all()
    assert set(customers["city"]) == {"Chennai"}


def test_legit_transactions_reference_valid_customers():
    customers = gen_customers(20)
    start, end = datetime(2026, 1, 1), datetime(2026, 2, 1)
    txns = gen_legit_transactions(customers, 500, start, end)
    assert len(txns) == 500
    assert set(txns["cc_num"]).issubset(set(customers["cc_num"]))
    assert (txns["amt"] > 0).all()
    assert txns["is_fraud"].eq(0).all()


def test_fraud_incidents_are_labeled_and_patterned():
    customers = gen_customers(20)
    start, end = datetime(2026, 1, 1), datetime(2026, 2, 1)
    fraud = gen_fraud_incidents(customers, 10, start, end)
    assert len(fraud) > 0
    assert fraud["is_fraud"].eq(1).all()
    assert set(fraud["fraud_pattern"]).issubset(
        {"card_testing", "geo_impossible_travel", "account_takeover_burst",
         "category_anomaly", "slow_skimming"}
    )


def test_enrich_computes_age_and_distance_without_nulls():
    customers = gen_customers(20)
    start, end = datetime(2026, 1, 1), datetime(2026, 2, 1)
    txns = gen_legit_transactions(customers, 200, start, end)
    enriched = enrich(txns, customers)
    assert enriched["age"].notna().all()
    assert enriched["distance_km"].notna().all()
    assert (enriched["age"] >= 0).all()
    assert (enriched["distance_km"] >= 0).all()
    assert "unix_time" in enriched.columns
    assert (enriched["unix_time"] > 1_700_000_000).all()  # sane 2023+ epoch seconds


def test_haversine_known_distance():
    # Chennai Central to Chennai Airport is roughly 18-20 km
    d = haversine_km(13.0827, 80.2707, 12.9941, 80.1709)
    assert 10 < d < 25
