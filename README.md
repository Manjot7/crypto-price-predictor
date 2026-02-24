# Crypto Price Direction Predictor

A real-time cryptocurrency analytics pipeline that streams live market data, processes it through a series of classical streaming and distributed algorithms, trains machine learning models, and serves predictions through a live web app — all running on a free cloud stack.

**Live demo:** [HuggingFace Space](https://https://huggingface.co/spaces/Manjot7/crypto-predictor)

---

## What it does

Every 60 seconds, an Oracle Cloud VM fetches OHLCV data for 20 cryptocurrency pairs from the Kraken API and publishes records to an Azure Service Bus queue. A consumer reads from that queue, applies four streaming algorithms in sequence, and writes processed batches to Azure Blob Storage. Kaggle notebooks then pull that data on demand to run MapReduce aggregations, LSH-based similarity search, and train a pair of Random Forest models with SHAP explainability. The trained models are saved back to Blob Storage and loaded automatically by a Gradio app on HuggingFace Spaces, which serves live predictions.

---

## Architecture

```
Kraken API (20 pairs, 60s intervals)
    │
    ▼
Producer.py  ──►  Azure Service Bus  ──►  Consumer.py
(Oracle VM)            queue                (Oracle VM)
                                                │
                        ┌───────────────────────┤
                        │   Streaming algorithms │
                        │   1. Bloom Filter      │  duplicate detection
                        │   2. Reservoir Sample  │  uniform stream sample (Algorithm R)
                        │   3. Differential Priv │  Laplace noise on sensitive fields
                        │   4. Flajolet-Martin   │  distinct count estimation
                        └───────────────────────┘
                                                │
                                                ▼
                                    Azure Blob Storage
                                    consumer_data/batch_*.json
                                    notebook_data/raw_stream.json
                                                │
                        ┌───────────────────────┘
                        │
               Python Notebooks
                        │
                        ├── Ingest_and_Stream: load data + streaming algorithm simulation
                        ├── MapReduce: mrjob MapReduce aggregation
                        ├── LSH: FAISS LSH similarity search
                        └── ML_and_SHAP: feature engineering + model training + SHAP
                                                │
                                    Azure Blob Storage
                                    models/clf.pkl
                                    models/reg.pkl
                                                │
                                                ▼
                                    HuggingFace Spaces
                                    Gradio app — live predictions
```

---

## Models

Two models are trained per run on 27 engineered features:

**Classifier** — predicts whether the next hourly close will be higher or lower than the current close (UP / DOWN). Evaluated with ROC-AUC, accuracy, and 5-fold cross-validation.

**Regressor** — predicts the percentage magnitude of that move. Evaluated with MAE, RMSE, and R².

### Features

| Category | Features |
|----------|---------|
| Candle anatomy | body size, range, upper/lower wick, body ratio, bullish flag |
| Momentum | % change over 1h, 3h, 6h, 24h |
| Moving averages | SMA 6/12/24/48, price deviation from each SMA |
| MACD | MACD line, signal line, histogram |
| Volatility | log-return std over 6h and 24h windows |
| Bollinger Bands | band width, %B position |
| RSI | 14-period RSI |
| Volume | 1h change, ratio to 24h average, normalised OBV |
| Liquidity | hourly volume as fraction of 24h total |

All features are constructed without look-ahead — the target is the next candle's close, and all features are derived from data available at prediction time.

---

## Streaming Algorithms

Implemented in `Consumer.py` and applied to every record as it arrives from the queue:

**Bloom Filter** — probabilistic set membership check for duplicate record IDs. 500,000 capacity, 1% false positive rate. Prevents duplicate candles from the same polling cycle entering the dataset.

**Reservoir Sampling (Algorithm R)** — maintains a uniform random sample of exactly 200 records from the stream without knowing the stream length in advance.

**Differential Privacy (Laplace Mechanism)** — adds calibrated Laplace noise to sensitive volume and market cap fields before storage. Noise scale is proportional to field magnitude with configurable epsilon (default ε=1.0).

**Flajolet-Martin** — estimates the number of distinct record IDs seen in the stream using 10 independent hash functions with a median estimator. Runs in O(1) space regardless of stream size.

---

## Distributed Processing

**MapReduce** (`MapReduce.ipynb`) — implemented with mrjob in local mode (no Hadoop cluster required). Two jobs run sequentially:
- Job A: per-symbol aggregation — average, min, max price, volatility, total volume
- Job B: hourly market-wide stats — market breadth (% of coins up), average price change, total volume

**LSH Similarity Search** (`LSH.ipynb`) — implemented with FAISS IndexLSH (no PySpark required). Builds a 13-feature vector per symbol and finds similar trading pairs based on volatility, trend, volume profile, and candle anatomy. Results are used in the Similar Coins tab of the web app.

---

## Web App

The Gradio app on HuggingFace Spaces loads models from Azure Blob at startup and serves four tabs:

- **Predict** — select any of the 20 pairs, get a direction prediction with confidence score and a SHAP waterfall chart explaining which features drove that prediction
- **Market Overview** — live 24h stats for all 20 pairs pulled directly from Kraken
- **Similar Coins** — LSH-based recommendations showing which pairs behave most similarly
- **About** — architecture overview, model metrics, feature list

---

## Stack

| Component | Service | Cost |
|-----------|---------|------|
| Market data | Kraken public API | Free, no key |
| Message broker | Azure Service Bus (Standard) | Free for 12 months |
| Object storage | Azure Blob Storage | Free for 12 months |
| VM (streaming) | Oracle Cloud Always Free | Free forever |
| ML compute | Kaggle Notebooks | Free (30hr/week) |
| App hosting | HuggingFace Spaces (Gradio) | Free forever |

---

## Repository Structure

```
├── Streaming/
│   ├── Producer.py                # Kraken → Azure Service Bus
│   └── Consumer.py                # Azure Service Bus → Azure Blob + streaming algorithms
├── Python Notebooks/
│   ├── Ingest_and_Stream.ipynb
│   ├── MapReduce.ipynb
│   ├── LSH.ipynb
│   └── ML_and_SHAP.ipynb
├── Huggingface Spaces/
│   ├── app.py
│   └── requirements.txt
└── Oracle Setup/
    ├── requirements.txt
    ├── crypto-producer.service    # systemd unit file
    └── crypto-consumer.service    # systemd unit file
```

---

## Setup

**Prerequisites:**
- Azure free account (azure.microsoft.com/free)
- Oracle Cloud free account (cloud.oracle.com)
- Kaggle account
- HuggingFace account

**Credentials required:**
- Azure Service Bus connection string → Producer.py, Consumer.py
- Azure Storage connection string → Consumer.py, Kaggle secrets, HuggingFace secrets

Once the Oracle VM services are running, data accumulates in Azure Blob automatically. Run the four Kaggle notebooks in order when ready to train, then restart the HuggingFace Space to load the new models.

---

## Retraining

Re-run ML Notebook at any time to retrain on all accumulated data, then go to HuggingFace Spaces → Settings → Factory reboot. The app will load the updated models automatically.
