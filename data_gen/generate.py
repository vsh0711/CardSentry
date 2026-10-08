"""
Synthetic but production-scale Chennai credit-card transaction generator for CardSentry.

Generates:
  - customers.parquet         ~5,000 synthetic Chennai cardholders
  - transactions.parquet      1-2M transactions over a 9-month window, ~1% fraud,
                               with realistic fraud patterns injected (not random noise):
                                 card_testing, geo_impossible_travel, account_takeover_burst,
                                 category_anomaly

Design notes:
  - Baseline ("legit") transactions are generated vectorized with numpy for speed.
  - Fraud patterns are generated per-incident (small loop, bounded by NUM_FRAUD_INCIDENTS)
    so each one is internally consistent (same card, tight time window, causally linked).
  - Written out in parquet chunks to data/raw/ so downstream jobs can read it the way a
    real data-engineering pipeline would (batched, columnar), not a single in-memory CSV.
"""
import argparse
import math
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from chennai_geo import (
    CHENNAI_LOCALITIES,
    OTHER_CITIES,
    CHENNAI_MERCHANTS,
    TAMIL_FIRST_NAMES_M,
    TAMIL_FIRST_NAMES_F,
    TAMIL_LAST_NAMES,
)

fake = Faker("en_IN")
RNG = np.random.default_rng(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
JOBS = [
    "Software Engineer", "Bank Manager", "Doctor", "Textile Merchant", "Teacher",
    "Chartered Accountant", "Auto Parts Dealer", "College Professor", "Civil Engineer",
    "Govt Employee (TN Secretariat)", "IT Consultant", "Restaurant Owner", "Lawyer",
    "Pharmacist", "Electrician", "Nurse", "HR Manager",
    "Marketing Executive", "Film Industry (Kollywood)", "Jewellery Trader",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def gen_customers(n_customers: int) -> pd.DataFrame:
    sexes = RNG.choice(["M", "F"], size=n_customers)
    firsts = [
        RNG.choice(TAMIL_FIRST_NAMES_M) if s == "M" else RNG.choice(TAMIL_FIRST_NAMES_F)
        for s in sexes
    ]
    lasts = RNG.choice(TAMIL_LAST_NAMES, size=n_customers)
    locality_idx = RNG.integers(0, len(CHENNAI_LOCALITIES), size=n_customers)
    jitter_lat = RNG.normal(0, 0.01, size=n_customers)
    jitter_lon = RNG.normal(0, 0.01, size=n_customers)

    rows = []
    for i in range(n_customers):
        loc_name, lat, lon = CHENNAI_LOCALITIES[locality_idx[i]]
        dob = fake.date_of_birth(minimum_age=19, maximum_age=72)
        rows.append({
            "cc_num": 4_000_000_000_000_000 + RNG.integers(0, 999_999_999),
            "first": firsts[i],
            "last": lasts[i],
            "gender": sexes[i],
            "locality": loc_name,
            "city": "Chennai",
            "state": "Tamil Nadu",
            "lat": lat + jitter_lat[i],
            "long": lon + jitter_lon[i],
            "job": RNG.choice(JOBS),
            "dob": dob.isoformat(),
            # a rough spend profile used to make each customer's baseline behaviour distinct
            "home_category_bias": RNG.choice(
                ["grocery", "food_dining", "fuel", "shopping_mall", "utility_bills"]
            ),
            "avg_txn_amt": float(RNG.lognormal(mean=6.5, sigma=0.6)),  # INR
        })
    return pd.DataFrame(rows)


def _merchant_arrays():
    names = np.array([m[0] for m in CHENNAI_MERCHANTS])
    cats = np.array([m[1] for m in CHENNAI_MERCHANTS])
    return names, cats


def gen_legit_transactions(customers: pd.DataFrame, n_txns: int, start: datetime, end: datetime) -> pd.DataFrame:
    merch_names, merch_cats = _merchant_arrays()
    n_cust = len(customers)
    cust_idx = RNG.integers(0, n_cust, size=n_txns)
    merch_idx = RNG.integers(0, len(merch_names), size=n_txns)

    span_seconds = int((end - start).total_seconds())
    # Poisson-process-like arrival: more transactions during 9am-9pm (Chennai retail hours)
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 1.4, 1.6, 1.8, 2.0,
        2.2, 2.0, 1.8, 1.8, 2.0, 2.2, 2.4, 2.2, 1.8, 1.2, 0.6, 0.3,
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = RNG.choice(24, size=n_txns, p=hour_weights)
    day_offsets = RNG.integers(0, max(span_seconds // 86400, 1), size=n_txns)
    minutes = RNG.integers(0, 60, size=n_txns)
    seconds = RNG.integers(0, 60, size=n_txns)
    timestamps = [
        start + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(day_offsets, hours, minutes, seconds)
    ]

    avg_amts = customers["avg_txn_amt"].to_numpy()[cust_idx]
    # wider per-transaction spread (sigma 0.5 -> 0.75) so legit amounts naturally
    # tail into the range fraud amounts also occupy -- real spend isn't this clean-cut
    amounts = np.clip(RNG.lognormal(mean=np.log(np.maximum(avg_amts, 50)), sigma=0.75, size=n_txns), 20, 150000)

    cust_lat = customers["lat"].to_numpy()[cust_idx]
    cust_lon = customers["long"].to_numpy()[cust_idx]
    jitter = RNG.normal(0, 0.03, size=(n_txns, 2))
    merch_lat = cust_lat + jitter[:, 0]
    merch_lon = cust_lon + jitter[:, 1]

    categories = merch_cats[merch_idx].copy()
    merchants = merch_names[merch_idx].copy()

    # hard negatives: ~3% are genuine big-ticket purchases in categories that otherwise
    # look fraud-like (jewellery/electronics/travel), at normal distance/hours -- these
    # overlap the fraud amount range on purpose so the model can't just threshold on amt+category
    hard_neg_mask = RNG.random(n_txns) < 0.03
    hard_neg_cats = np.array(["jewellery", "electronics", "travel"])
    hard_neg_idx = np.where(hard_neg_mask)[0]
    if len(hard_neg_idx):
        chosen_cats = RNG.choice(hard_neg_cats, size=len(hard_neg_idx))
        for i, cat in zip(hard_neg_idx, chosen_cats):
            pool = np.where(merch_cats == cat)[0]
            m = pool[RNG.integers(0, len(pool))]
            categories[i] = merch_cats[m]
            merchants[i] = merch_names[m]
            amounts[i] = float(np.clip(RNG.lognormal(mean=9.2, sigma=0.7), 5000, 140000))

    df = pd.DataFrame({
        "cc_num": customers["cc_num"].to_numpy()[cust_idx],
        "trans_num": [uuid.uuid4().hex for _ in range(n_txns)],
        "trans_time": timestamps,
        "category": categories,
        "merchant": merchants,
        "amt": np.round(amounts, 2),
        "merch_lat": merch_lat,
        "merch_long": merch_lon,
        "is_fraud": 0,
        "fraud_pattern": "none",
    })
    return df


def gen_legit_bursts(customers: pd.DataFrame, n_bursts: int, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Genuine 'shopping trip' bursts: a customer visits several merchants within
    30-90 minutes (mall trip, errands day). This deliberately mimics the temporal
    clustering that the fraud patterns also produce, so velocity features alone
    can't separate fraud from legit -- the model has to use amount/category/distance
    shape too, which is more realistic.
    """
    merch_names, merch_cats = _merchant_arrays()
    rows = []
    for _ in range(n_bursts):
        cust = customers.iloc[RNG.integers(0, len(customers))]
        base_time = start + timedelta(seconds=int(RNG.integers(0, int((end - start).total_seconds()))))
        k = RNG.integers(2, 6)
        for j in range(k):
            m_idx = RNG.integers(0, len(merch_names))
            avg_amt = max(float(cust.avg_txn_amt), 50)
            rows.append({
                "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex,
                "trans_time": base_time + timedelta(minutes=int(j * RNG.integers(5, 25))),
                "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                "amt": round(float(np.clip(RNG.lognormal(np.log(avg_amt), 0.6), 20, 20000)), 2),
                "merch_lat": cust.lat + RNG.normal(0, 0.03), "merch_long": cust.long + RNG.normal(0, 0.03),
                "is_fraud": 0, "fraud_pattern": "none",
            })
    return pd.DataFrame(rows)


def gen_fraud_incidents(customers: pd.DataFrame, n_incidents: int, start: datetime, end: datetime) -> pd.DataFrame:
    merch_names, merch_cats = _merchant_arrays()
    jewellery_idx = np.where(merch_cats == "jewellery")[0]
    electronics_idx = np.where(merch_cats == "electronics")[0]
    pattern_choices = [
        "card_testing", "geo_impossible_travel", "account_takeover_burst",
        "category_anomaly", "slow_skimming",
    ]
    pattern_weights = [0.22, 0.18, 0.18, 0.12, 0.30]  # slow_skimming is the hardest to catch -> most common
    all_rows = []

    for _ in range(n_incidents):
        cust = customers.iloc[RNG.integers(0, len(customers))]
        pattern = RNG.choice(pattern_choices, p=pattern_weights)
        base_time = start + timedelta(seconds=int(RNG.integers(0, int((end - start).total_seconds()))))

        if pattern == "card_testing":
            k = RNG.integers(5, 16)
            for j in range(k):
                m_idx = RNG.integers(0, len(merch_names))
                all_rows.append({
                    "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex,
                    "trans_time": base_time + timedelta(seconds=int(j * RNG.integers(10, 90))),
                    "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                    # widened upper bound so it overlaps with ordinary small purchases
                    "amt": round(float(RNG.uniform(1, 400)), 2),
                    "merch_lat": cust.lat + RNG.normal(0, 0.02), "merch_long": cust.long + RNG.normal(0, 0.02),
                    "is_fraud": 1, "fraud_pattern": pattern,
                })

        elif pattern == "geo_impossible_travel":
            m_idx = RNG.integers(0, len(merch_names))
            all_rows.append({
                "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex, "trans_time": base_time,
                "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                "amt": round(float(RNG.uniform(200, 5000)), 2),
                "merch_lat": cust.lat, "merch_long": cust.long,
                "is_fraud": 1, "fraud_pattern": pattern,
            })
            other_city, c_lat, c_lon = OTHER_CITIES[RNG.integers(0, len(OTHER_CITIES))]
            # widened gap so some "impossible" trips are merely improbable, not physically impossible
            gap_minutes = RNG.integers(5, 150)
            m_idx2 = RNG.integers(0, len(merch_names))
            all_rows.append({
                "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex,
                "trans_time": base_time + timedelta(minutes=int(gap_minutes)),
                "category": merch_cats[m_idx2], "merchant": f"{merch_names[m_idx2]} ({other_city})",
                "amt": round(float(RNG.uniform(500, 20000)), 2),
                "merch_lat": c_lat + RNG.normal(0, 0.05), "merch_long": c_lon + RNG.normal(0, 0.05),
                "is_fraud": 1, "fraud_pattern": pattern,
            })

        elif pattern == "account_takeover_burst":
            k = RNG.integers(3, 7)
            pool = jewellery_idx if RNG.random() < 0.5 else electronics_idx
            for j in range(k):
                m_idx = pool[RNG.integers(0, len(pool))]
                all_rows.append({
                    "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex,
                    "trans_time": base_time + timedelta(minutes=int(j * RNG.integers(2, 10))),
                    "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                    # lowered floor so it overlaps with the hard-negative legit big-ticket purchases
                    "amt": round(float(RNG.uniform(3000, 95000)), 2),
                    "merch_lat": cust.lat + RNG.normal(0, 0.03), "merch_long": cust.long + RNG.normal(0, 0.03),
                    "is_fraud": 1, "fraud_pattern": pattern,
                })

        elif pattern == "category_anomaly":
            pool = jewellery_idx if RNG.random() < 0.5 else electronics_idx
            m_idx = pool[RNG.integers(0, len(pool))]
            # not always an odd hour anymore -- some category anomalies happen in broad daylight
            hour = int(RNG.integers(1, 4)) if RNG.random() < 0.6 else int(RNG.integers(9, 21))
            ts = base_time.replace(hour=hour)
            all_rows.append({
                "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex, "trans_time": ts,
                "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                "amt": round(float(RNG.uniform(5000, 120000)), 2),
                "merch_lat": cust.lat + RNG.normal(0, 0.01), "merch_long": cust.long + RNG.normal(0, 0.01),
                "is_fraud": 1, "fraud_pattern": pattern,
            })

        else:  # slow_skimming -- a cautious fraudster spends small, ordinary-looking
               # amounts spread over hours so velocity features barely move; the
               # hardest pattern to catch, deliberately overlapping legit behaviour
            k = RNG.integers(3, 7)
            for j in range(k):
                m_idx = RNG.integers(0, len(merch_names))
                all_rows.append({
                    "cc_num": cust.cc_num, "trans_num": uuid.uuid4().hex,
                    "trans_time": base_time + timedelta(minutes=int(j * RNG.integers(40, 240))),
                    "category": merch_cats[m_idx], "merchant": merch_names[m_idx],
                    "amt": round(float(np.clip(RNG.lognormal(mean=6.3, sigma=0.6), 50, 4000)), 2),
                    "merch_lat": cust.lat + RNG.normal(0, 0.1), "merch_long": cust.long + RNG.normal(0, 0.1),
                    "is_fraud": 1, "fraud_pattern": pattern,
                })

    return pd.DataFrame(all_rows)


def enrich(df: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(
        customers[["cc_num", "first", "last", "dob", "lat", "long"]],
        on="cc_num", how="left", suffixes=("", "_home"),
    )
    df["trans_time"] = pd.to_datetime(df["trans_time"])
    df["unix_time"] = df["trans_time"].values.astype("datetime64[s]").astype("int64")
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((df["trans_time"] - df["dob"]).dt.days / 365.25).round(1)
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]).round(2)
    df["trans_date"] = df["trans_time"].dt.date.astype(str)
    df["trans_time_str"] = df["trans_time"].dt.strftime("%H:%M:%S")
    df = df.drop(columns=["lat", "long", "dob"])
    return df.sort_values("unix_time").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--transactions", type=int, default=1_500_000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--months", type=int, default=9)
    parser.add_argument("--out", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = datetime(2026, 10, 8)
    start = end - timedelta(days=30 * args.months)

    print(f"Generating {args.customers} customers...")
    customers = gen_customers(args.customers)
    customers.to_parquet(out_dir / "customers.parquet", index=False)

    n_fraud_target = int(args.transactions * args.fraud_rate)
    # each incident yields ~1-8 rows; approximate incident count accordingly
    n_incidents = max(1, n_fraud_target // 4)
    n_bursts = max(1, int(args.transactions * 0.015 / 3.5))  # ~1.5% of volume in legit bursts
    n_legit = args.transactions - n_fraud_target

    print(f"Generating ~{n_legit:,} legit transactions (incl. hard-negative big purchases)...")
    legit = gen_legit_transactions(customers, n_legit, start, end)

    print(f"Generating {n_bursts:,} legit 'shopping trip' bursts (temporal-clustering noise)...")
    bursts = gen_legit_bursts(customers, n_bursts, start, end)

    print(f"Generating {n_incidents:,} fraud incidents across 5 patterns (~{n_fraud_target:,} rows)...")
    fraud = gen_fraud_incidents(customers, n_incidents, start, end)

    print("Merging, enriching (age/distance), sorting by time...")
    full = pd.concat([legit, bursts, fraud], ignore_index=True)
    full = enrich(full, customers)

    # label noise: real fraud labels always have some error (chargebacks reported late,
    # disputes that turn out to be the cardholder's own forgotten purchase, etc.)
    noise_rng = np.random.default_rng(7)
    fraud_mask = full["is_fraud"] == 1
    flip_fraud_to_legit = fraud_mask & (noise_rng.random(len(full)) < 0.03)
    flip_legit_to_fraud = (~fraud_mask) & (noise_rng.random(len(full)) < 0.002)
    full.loc[flip_fraud_to_legit, "is_fraud"] = 0
    full.loc[flip_legit_to_fraud, "is_fraud"] = 1
    n_flipped = int(flip_fraud_to_legit.sum() + flip_legit_to_fraud.sum())
    print(f"Applied label noise: flipped {n_flipped:,} labels "
          f"({int(flip_fraud_to_legit.sum())} fraud->legit, {int(flip_legit_to_fraud.sum())} legit->fraud)")

    actual_fraud_rate = full["is_fraud"].mean()
    print(f"Total rows: {len(full):,} | actual fraud rate: {actual_fraud_rate:.4%}")
    print(f"Fraud pattern mix:\n{full[full.is_fraud==1].fraud_pattern.value_counts()}")

    # write in a handful of time-ordered chunks, like a real batch pipeline would
    n_chunks = 10
    chunk_size = math.ceil(len(full) / n_chunks)
    for i in range(n_chunks):
        chunk = full.iloc[i * chunk_size: (i + 1) * chunk_size]
        if len(chunk) == 0:
            continue
        chunk.to_parquet(out_dir / f"transactions_part_{i:02d}.parquet", index=False)

    print(f"Done. Wrote customers.parquet + {n_chunks} transaction part files to {out_dir}")


if __name__ == "__main__":
    main()
