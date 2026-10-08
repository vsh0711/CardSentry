"""
Feast feature definitions for CardSentry.

Demonstrates the actual "train/serve parity via a feature store" pattern from the
project brief: the same engineered features (features/engineer.py) are written to
an offline parquet source here, registered with Feast, and retrievable either as a
point-in-time-correct historical training set (get_historical_features) or as
low-latency online lookups (get_online_features) after materialization -- so a
training job and a serving job can never silently drift apart.

Apply with:  cd feature_repo && feast apply
"""
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

card = Entity(name="cc_num", join_keys=["cc_num"], value_type=ValueType.STRING)

card_txn_source = FileSource(
    name="card_transaction_features_source",
    path="data/card_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

card_transaction_features = FeatureView(
    name="card_transaction_features",
    entities=[card],
    ttl=timedelta(days=2),
    schema=[
        Field(name="amt", dtype=Float32),
        Field(name="distance_km", dtype=Float32),
        Field(name="txn_count_10m", dtype=Float32),
        Field(name="txn_count_1h", dtype=Float32),
        Field(name="txn_count_24h", dtype=Float32),
        Field(name="amt_sum_1h", dtype=Float32),
        Field(name="amt_mean_24h", dtype=Float32),
        Field(name="time_since_last_txn_sec", dtype=Float32),
        Field(name="distance_from_last_txn_km", dtype=Float32),
        Field(name="hour_of_day", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_night", dtype=Int64),
    ],
    online=True,
    source=card_txn_source,
)
