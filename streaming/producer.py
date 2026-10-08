"""
Kafka producer for CardSentry.

Streams synthetic Chennai card transactions onto the `cardsentry-transactions`
topic at a configurable rate (default mirrors the original project's claim of
~3,000 events/sec, tunable down for a live free-tier demo). Points at Redpanda
Serverless (or any Kafka-compatible broker) via env vars:

  KAFKA_BOOTSTRAP_SERVERS, KAFKA_API_KEY, KAFKA_API_SECRET  (SASL_SSL)
  or just KAFKA_BOOTSTRAP_SERVERS for a local/no-auth broker.
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from confluent_kafka import Producer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_gen.chennai_geo import CHENNAI_LOCALITIES, CHENNAI_MERCHANTS

TOPIC = os.environ.get("KAFKA_TOPIC", "cardsentry-transactions")
RNG = np.random.default_rng()


def build_kafka_config() -> dict:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    conf = {"bootstrap.servers": bootstrap, "linger.ms": 20, "batch.size": 65536}
    api_key = os.environ.get("KAFKA_API_KEY")
    api_secret = os.environ.get("KAFKA_API_SECRET")
    if api_key and api_secret:
        conf.update({
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": os.environ.get("KAFKA_SASL_MECHANISM", "SCRAM-SHA-256"),
            "sasl.username": api_key,
            "sasl.password": api_secret,
        })
    return conf


def load_customer_pool():
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "customers.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def random_transaction(customers: pd.DataFrame | None) -> dict:
    merch_name, category = CHENNAI_MERCHANTS[RNG.integers(0, len(CHENNAI_MERCHANTS))]
    now = datetime.utcnow()

    if customers is not None and len(customers):
        cust = customers.iloc[RNG.integers(0, len(customers))]
        cc_num, first, last = int(cust.cc_num), cust["first"], cust["last"]
        home_lat, home_lon = float(cust.lat), float(cust["long"])
        locality = cust.locality
        dob = pd.to_datetime(cust.dob)
        age = round((now - dob.to_pydatetime().replace(tzinfo=None)).days / 365.25, 1)
    else:
        locality, home_lat, home_lon = CHENNAI_LOCALITIES[RNG.integers(0, len(CHENNAI_LOCALITIES))]
        cc_num = 4_000_000_000_000_000 + int(RNG.integers(0, 999_999_999))
        first, last, age = "Guest", "Customer", 35.0

    # occasionally inject an impossible-travel style fraud probe for a livelier demo
    if RNG.random() < 0.01:
        merch_lat = home_lat + RNG.normal(0, 2.0)
        merch_lon = home_lon + RNG.normal(0, 2.0)
        amt = float(np.clip(RNG.lognormal(mean=9.5, sigma=0.5), 500, 150000))
    else:
        merch_lat = home_lat + RNG.normal(0, 0.03)
        merch_lon = home_lon + RNG.normal(0, 0.03)
        amt = float(np.clip(RNG.lognormal(mean=6.8, sigma=0.6), 20, 150000))

    distance_km = round(float(haversine_km(home_lat, home_lon, merch_lat, merch_lon)), 2)

    return {
        "trans_num": uuid.uuid4().hex,
        "cc_num": cc_num,
        "first": first,
        "last": last,
        "locality": locality,
        "merchant": merch_name,
        "category": category,
        "amt": round(amt, 2),
        "merch_lat": merch_lat,
        "merch_long": merch_lon,
        "distance_km": distance_km,
        "age": age,
        "unix_time": now.timestamp(),
    }


def main():
    rate = float(os.environ.get("PRODUCER_EVENTS_PER_SEC", "20"))
    total = int(os.environ.get("PRODUCER_TOTAL_EVENTS", "0"))  # 0 = run forever

    producer = Producer(build_kafka_config())
    customers = load_customer_pool()
    print(f"Producing to topic '{TOPIC}' at {rate} events/sec "
          f"({'forever' if total == 0 else f'{total} events'})...")

    sent = 0
    interval = 1.0 / rate if rate > 0 else 0
    try:
        while total == 0 or sent < total:
            txn = random_transaction(customers)
            producer.produce(TOPIC, key=str(txn["cc_num"]), value=json.dumps(txn))
            producer.poll(0)
            sent += 1
            if sent % 200 == 0:
                producer.flush(2)
                print(f"  sent {sent:,} events")
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(5)
        print(f"Done. Sent {sent:,} events total.")


if __name__ == "__main__":
    main()
