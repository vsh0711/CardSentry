"""
Kafka consumer for CardSentry: consumes transactions from `cardsentry-transactions`,
scores each one with the trained ensemble (api/scorer.py), and persists the result
to the shared TransactionStore so the Streamlit dashboard shows it in real time.

This is the "Spark Streaming job" equivalent from the original project, replaced
with a lightweight Python consumer running the same model code as the FastAPI
service (api/scorer.py) -- one scoring implementation, two entry points.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

from confluent_kafka import Consumer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.scorer import FraudScorer
from api.store import TransactionStore

TOPIC = os.environ.get("KAFKA_TOPIC", "cardsentry-transactions")


def build_kafka_config() -> dict:
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    conf = {
        "bootstrap.servers": bootstrap,
        "group.id": os.environ.get("KAFKA_GROUP_ID", "cardsentry-scoring-consumer"),
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }
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


def run_consumer_loop(scorer: FraudScorer, store: TransactionStore, stop_event: threading.Event | None = None):
    """
    The actual consume-score-persist loop, factored out so it can run either as
    the `__main__` entrypoint (standalone `python streaming/consumer.py`) or as a
    background thread inside the FastAPI process (api/main.py) -- the latter lets
    a single free-tier Render web service do both HTTP scoring and continuous
    Kafka consumption, since Render's free tier doesn't offer background workers.
    """
    consumer = Consumer(build_kafka_config())
    consumer.subscribe([TOPIC])
    print(f"Consuming '{TOPIC}' and scoring transactions in real time...")

    processed = 0
    last_report = time.time()
    try:
        while stop_event is None or not stop_event.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print("Kafka error:", msg.error())
                continue

            txn = json.loads(msg.value())
            import datetime
            dt = datetime.datetime.fromtimestamp(txn["unix_time"])
            txn["hour_of_day"] = dt.hour
            txn["day_of_week"] = dt.weekday()

            result = scorer.score_transaction(txn)
            record = {**txn, **result, "scored_at": time.time()}
            store.append(record)
            processed += 1

            if result["is_fraud_predicted"]:
                print(f"  🚨 FRAUD  cc_num=...{str(txn['cc_num'])[-4:]} amt=₹{txn['amt']:.2f} "
                      f"merchant={txn['merchant']} prob={result['fraud_probability']:.3f}")

            if time.time() - last_report > 10:
                print(f"  processed {processed:,} transactions so far "
                      f"(avg latency {result['latency_ms']:.1f} ms)")
                last_report = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        print(f"Consumer stopped. Processed {processed:,} transactions.")


def main():
    scorer = FraudScorer()
    store = TransactionStore()
    run_consumer_loop(scorer, store)


if __name__ == "__main__":
    main()
