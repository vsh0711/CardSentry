"""
Quick local traffic simulator that POSTs transactions straight to the FastAPI
/score endpoint (no Kafka needed) -- useful for demoing the dashboard without
provisioning a broker, and for the test suite / local sanity checks.
"""
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_gen.chennai_geo import CHENNAI_MERCHANTS

API_URL = "http://localhost:8000/score"
RNG = np.random.default_rng()


def load_customers():
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "customers.parquet"
    return pd.read_parquet(path) if path.exists() else None


def make_txn(customers, force_fraud=False):
    merch_name, category = random.choice(CHENNAI_MERCHANTS)
    if customers is not None and len(customers):
        cust = customers.iloc[RNG.integers(0, len(customers))]
        cc_num, first, last, locality = int(cust.cc_num), cust["first"], cust["last"], cust.locality
        home_lat, home_lon, age = float(cust.lat), float(cust["long"]), 35.0
    else:
        cc_num, first, last, locality, age = 4000000000000001, "Guest", "User", "Chennai", 30.0
        home_lat, home_lon = 13.0827, 80.2707

    if force_fraud:
        amt = float(RNG.uniform(20000, 120000))
        merch_lat = home_lat + RNG.normal(0, 2.0)
        merch_lon = home_lon + RNG.normal(0, 2.0)
        category = "jewellery"
    else:
        amt = float(np.clip(RNG.lognormal(6.8, 0.5), 20, 5000))
        merch_lat = home_lat + RNG.normal(0, 0.03)
        merch_lon = home_lon + RNG.normal(0, 0.03)

    distance_km = float(_haversine_km(home_lat, home_lon, merch_lat, merch_lon))

    return {
        "cc_num": cc_num, "first": first, "last": last, "merchant": merch_name,
        "category": category, "amt": round(amt, 2), "locality": locality,
        "distance_km": round(distance_km, 2), "age": age,
        "merch_lat": merch_lat, "merch_long": merch_lon,
    }


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main(n=300, fraud_ratio=0.02, delay=0.05):
    customers = load_customers()
    with httpx.Client(timeout=5.0) as client:
        for i in range(n):
            force_fraud = random.random() < fraud_ratio
            txn = make_txn(customers, force_fraud)
            r = client.post(API_URL, json=txn)
            r.raise_for_status()
            result = r.json()
            tag = "FRAUD" if result["is_fraud_predicted"] else "ok"
            print(f"[{i+1}/{n}] {tag:5s} prob={result['fraud_probability']:.3f} "
                  f"amt=₹{txn['amt']:.0f} {txn['merchant']}")
            time.sleep(delay)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--fraud-ratio", type=float, default=0.02)
    p.add_argument("--delay", type=float, default=0.05)
    args = p.parse_args()
    main(args.n, args.fraud_ratio, args.delay)
