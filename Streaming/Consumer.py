import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from azure.servicebus import ServiceBusClient
from azure.storage.blob import BlobServiceClient
from pybloom_live import BloomFilter

# Logging
LOG_DIR = "/home/ubuntu/crypto_stream/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/consumer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("consumer")

# Azure Config
AZURE_SERVICEBUS_CONN_STR  = " "
AZURE_STORAGE_CONN_STR     = " "
AZURE_CONTAINER_NAME       = " "
QUEUE_NAME                 = " "

BLOB_PREFIX        = "consumer_data/"
SAMPLE_SIZE        = 200
BATCH_SIZE_TARGET  = 200
EPSILON            = 1.0

# Azure Blob Storage client
try:
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN_STR)
    container    = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    log.info(f"Azure Blob Storage connected → {AZURE_CONTAINER_NAME}")
except Exception as e:
    log.error(f"Azure Blob Storage init failed: {e}")
    sys.exit(1)

# Graceful shutdown
_running = True

def _handle_signal(sig, frame):
    global _running
    log.info(f"Caught signal {sig} — finishing current batch then stopping...")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# Streaming Algorithms

# Bloom Filter
bloom   = BloomFilter(capacity=500_000, error_rate=0.01)
n_dupes = 0

def bloom_check(record_id: str) -> bool:
    global n_dupes
    if record_id in bloom:
        n_dupes += 1
        return True
    bloom.add(record_id)
    return False


# Reservoir Sampling
reservoir: list = []
n_seen          = 0

def reservoir_add(item: dict) -> None:
    global n_seen, reservoir
    n_seen += 1
    if len(reservoir) < SAMPLE_SIZE:
        reservoir.append(item)
    else:
        j = random.randint(0, n_seen - 1)
        if j < SAMPLE_SIZE:
            reservoir[j] = item


# Differential Privacy
SENSITIVE_FIELDS = ["volume_24h", "quote_volume_24h", "volume", "quote_volume"]

def add_dp_noise(record: dict, epsilon: float = EPSILON) -> dict:
    noisy = record.copy()
    for field in SENSITIVE_FIELDS:
        if field in noisy and noisy[field] is not None:
            val          = float(noisy[field])
            sensitivity  = max(abs(val) * 0.001, 1e-6)
            noise        = np.random.laplace(0, sensitivity / epsilon)
            noisy[field] = val + noise
    return noisy


# Flajolet-Martin Distinct Count
_fm_max_zeros = [0] * 10

def fm_add(record_id: str) -> None:
    for i in range(10):
        h     = hashlib.sha256(f"{i}:{record_id}".encode()).hexdigest()
        bits  = bin(int(h, 16))[2:]
        zeros = len(bits) - len(bits.rstrip("0"))
        _fm_max_zeros[i] = max(_fm_max_zeros[i], zeros)

def fm_estimate() -> int:
    return int(np.median([2 ** z for z in _fm_max_zeros]))


# Azure Blob Storage Upload

def upload_batch(batch: list, batch_num: int) -> str:
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid      = uuid4().hex[:6]
    blob_name = f"{BLOB_PREFIX}batch_{ts}_{uid}.json"

    payload = {
        "batch_number":  batch_num,
        "upload_time":   datetime.now(timezone.utc).isoformat(),
        "record_count":  len(batch),
        "stream_stats": {
            "total_seen":          n_seen,
            "duplicates_detected": n_dupes,
            "estimated_distinct":  fm_estimate(),
            "epsilon":             EPSILON,
        },
        "records": batch,
    }

    container.upload_blob(
        name=blob_name,
        data=json.dumps(payload, indent=2),
        overwrite=True,
        content_settings=None,
    )
    uri = f"https://{blob_service.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{blob_name}"
    log.info(f"[Blob] Uploaded {len(batch)} records → {blob_name}")
    return uri


def rebuild_raw_stream_json() -> None:
    """
    Merge all batch blobs into a single notebook_data/raw_stream.json blob.
    This is what all Kaggle notebooks pull — one flat deduplicated list.
    """
    try:
        blobs = list(container.list_blobs(name_starts_with=BLOB_PREFIX))
        if not blobs:
            return

        all_records = []
        for blob in blobs:
            try:
                data    = container.download_blob(blob.name).readall()
                content = json.loads(data)
                all_records.extend(content.get("records", []))
            except Exception as e:
                log.warning(f"Could not read {blob.name}: {e}")

        # Deduplicate by record ID
        seen   = {}
        for r in all_records:
            seen[r.get("id", "")] = r
        deduped = list(seen.values())

        container.upload_blob(
            name="notebook_data/raw_stream.json",
            data=json.dumps(deduped),
            overwrite=True,
        )
        log.info(f"[Blob] Rebuilt notebook_data/raw_stream.json → "
                 f"{len(deduped):,} records from {len(blobs)} batches")

    except Exception as e:
        log.error(f"rebuild_raw_stream_json failed: {e}")


# Service Bus Consumer

def run():
    global reservoir, n_seen

    log.info(f"Consumer starting | Queue: {QUEUE_NAME} | Batch target: {BATCH_SIZE_TARGET}")

    batch_num          = 0
    total_uploaded     = 0
    consecutive_errors = 0

    with ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONN_STR) as sb_client:
        with sb_client.get_queue_receiver(queue_name=QUEUE_NAME, max_wait_time=5) as receiver:
            log.info("Connected to Azure Service Bus — waiting for messages...")

            while _running:
                try:
                    messages = receiver.receive_messages(
                        max_message_count=50,
                        max_wait_time=5,
                    )

                    if not messages:
                        if n_seen % 200 == 0 and n_seen > 0:
                            log.info(f"Waiting... reservoir={len(reservoir)} seen={n_seen}")
                        time.sleep(3)
                        continue

                    for msg in messages:
                        try:
                            record = json.loads(str(msg))
                            rid    = record.get("id", str(uuid4()))

                            bloom_check(rid)               # Bloom Filter
                            fm_add(rid)                    # Flajolet-Martin
                            record = add_dp_noise(record)  # Differential Privacy
                            reservoir_add(record)          # Reservoir Sampling

                            receiver.complete_message(msg)  # ACK — remove from queue

                        except Exception as e:
                            log.warning(f"Error processing message: {e}")
                            receiver.abandon_message(msg)   # NACK — return to queue

                    # Upload batch when reservoir is full
                    if len(reservoir) >= BATCH_SIZE_TARGET:
                        batch_num += 1
                        upload_batch(list(reservoir), batch_num)
                        total_uploaded += len(reservoir)
                        log.info(
                            f"Batch {batch_num} done | Total: {total_uploaded} | "
                            f"Dupes: {n_dupes} | ~Distinct: {fm_estimate()}"
                        )
                        rebuild_raw_stream_json()
                        reservoir = []
                        n_seen    = 0

                    consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    log.error(f"Error (#{consecutive_errors}): {e}")
                    if consecutive_errors >= 10:
                        log.error("Too many errors. Exiting for systemd restart.")
                        sys.exit(1)
                    time.sleep(15)

    log.info(f"Consumer stopped. Batches: {batch_num}, records: {total_uploaded}")


if __name__ == "__main__":
    run()
