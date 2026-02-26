import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import requests
from azure.servicebus import ServiceBusClient, ServiceBusMessage

# Logging
LOG_DIR = "/home/ubuntu/crypto_stream/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/producer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("producer")

# Azure config
AZURE_SERVICEBUS_CONN_STR = ""
QUEUE_NAME                = ""

KRAKEN_OHLC   = "https://api.kraken.com/0/public/OHLC"
KRAKEN_TICKER = "https://api.kraken.com/0/public/Ticker"
POLL_INTERVAL = 60   # seconds between fetch cycles

SYMBOLS = {
    "XBTUSD":  "BTC",
    "ETHUSD":  "ETH",
    "SOLUSD":  "SOL",
    "XRPUSD":  "XRP",
    "ADAUSD":  "ADA",
    "DOTUSD":  "DOT",
    "DOGEUSD": "DOGE",
    "LTCUSD":  "LTC",
    "AVAXUSD": "AVAX",
    "LINKUSD": "LINK",
    "UNIUSD":  "UNI",
    "ATOMUSD": "ATOM",
    "XLMUSD":  "XLM",
    "FILUSD":  "FIL",
    "ICPUSD":  "ICP",
    "ETCUSD":  "ETC",
    "ALGOUSD": "ALGO",
    "TRXUSD":  "TRX",
    "AAVEUSD": "AAVE",
    "NEARUSD": "NEAR",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "crypto-producer/1.0"})

# Graceful shutdown
_running = True

def _handle_signal(sig, frame):
    global _running
    log.info(f"Caught signal {sig} — shutting down after current cycle...")
    _running = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# Kraken fetching

def fetch_kraken_ohlc(pair: str) -> list:
    try:
        r = SESSION.get(KRAKEN_OHLC, params={"pair": pair, "interval": 60}, timeout=15)
        r.raise_for_status()
        data    = r.json()
        if data.get("error"):
            log.warning(f"Kraken OHLC error [{pair}]: {data['error']}")
            return []
        result  = data.get("result", {})
        candles = next((v for k, v in result.items() if k != "last"), [])
        return candles[-24:]
    except Exception as e:
        log.warning(f"Kraken OHLC failed [{pair}]: {e}")
        return []


def fetch_kraken_ticker(pair: str) -> dict:
    try:
        r = SESSION.get(KRAKEN_TICKER, params={"pair": pair}, timeout=10)
        r.raise_for_status()
        data   = r.json()
        result = data.get("result", {})
        return next(iter(result.values()), {})
    except Exception:
        return {}


def build_record(pair: str, base: str, candles: list, ticker: dict) -> dict:
    latest      = candles[-1]
    last_price  = float(ticker.get("c", [0])[0]) if ticker else float(latest[4])
    volume_24h  = float(ticker.get("v", [0, 0])[1]) if ticker else 0
    high_24h    = float(ticker.get("h", [0, 0])[1]) if ticker else 0
    low_24h     = float(ticker.get("l", [0, 0])[1]) if ticker else 0
    open_24h    = float(ticker.get("o", last_price)) if ticker else last_price
    pct_chg_24h = (last_price - open_24h) / open_24h * 100 if open_24h else 0
    candle_ts   = int(latest[0])
    poll_ts     = int(datetime.now(timezone.utc).timestamp())
    return {
        "id":                   f"{pair}_{candle_ts}_{poll_ts}",
        "symbol":               pair,
        "base_asset":           base,
        "timestamp":            datetime.fromtimestamp(candle_ts, tz=timezone.utc).isoformat(),
        "open_time":            candle_ts * 1000,
        "open":                 float(latest[1]),
        "high":                 float(latest[2]),
        "low":                  float(latest[3]),
        "close":                float(latest[4]),
        "vwap":                 float(latest[5]),
        "volume":               float(latest[6]),
        "trade_count":          int(latest[7]),
        "quote_volume":         float(latest[6]) * float(latest[5]),
        "current_price":        last_price,
        "price_change_pct_24h": round(pct_chg_24h, 4),
        "high_24h":             high_24h,
        "low_24h":              low_24h,
        "volume_24h":           volume_24h,
        "quote_volume_24h":     volume_24h * last_price,
        "market_cap":           0,
        "circulating_supply":   0,
        "max_supply":           None,
        "rank":                 0,
    }


# Main loop

def run():
    log.info(f"Producer starting | Queue: {QUEUE_NAME} | {len(SYMBOLS)} symbols (Kraken)")

    consecutive_errors = 0

    with ServiceBusClient.from_connection_string(AZURE_SERVICEBUS_CONN_STR) as sb_client:
        with sb_client.get_queue_sender(queue_name=QUEUE_NAME) as sender:
            log.info("Connected to Azure Service Bus")

            while _running:
                cycle_start = time.time()
                log.info("--- Fetch cycle start ---")
                messages = []

                for pair, base in SYMBOLS.items():
                    if not _running:
                        break
                    candles = fetch_kraken_ohlc(pair)
                    if not candles:
                        continue
                    ticker = fetch_kraken_ticker(pair)
                    record = build_record(pair, base, candles, ticker)
                    messages.append(ServiceBusMessage(json.dumps(record)))
                    time.sleep(0.5)

                if messages:
                    try:
                        # Send all messages in one batch for efficiency
                        batch = sender.create_message_batch()
                        for msg in messages:
                            try:
                                batch.add_message(msg)
                            except Exception:
                                # Batch full — send current batch and start new one
                                sender.send_messages(batch)
                                batch = sender.create_message_batch()
                                batch.add_message(msg)
                        sender.send_messages(batch)
                        consecutive_errors = 0
                        elapsed = time.time() - cycle_start
                        log.info(f"Sent {len(messages)}/{len(SYMBOLS)} records in {elapsed:.1f}s")
                    except Exception as e:
                        consecutive_errors += 1
                        log.error(f"Send error (#{consecutive_errors}): {e}")
                        if consecutive_errors >= 10:
                            log.error("Too many errors. Exiting for systemd restart.")
                            sys.exit(1)
                        time.sleep(30)

                elapsed = time.time() - cycle_start
                sleep   = max(0, POLL_INTERVAL - elapsed)
                if _running and sleep > 0:
                    log.info(f"Sleeping {sleep:.0f}s until next cycle...")
                    time.sleep(sleep)

    log.info("Producer shut down cleanly.")


if __name__ == "__main__":
    run()
