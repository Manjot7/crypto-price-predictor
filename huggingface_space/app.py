# =============================================================================
# app.py — HuggingFace Spaces (Gradio)
# =============================================================================
# DEPLOY STEPS:
#   1. huggingface.co → New Space → SDK: Gradio → create
#   2. Upload this file as app.py
#   3. Upload requirements.txt
#   4. Settings → Variables and secrets → add:
#        AZURE_STORAGE_CONN_STR = your Azure Storage connection string
#        AZURE_CONTAINER        = crypto-data
#   5. Space auto-builds → live forever, no PC needed
#
# Models load from Azure Blob automatically. After retraining in Notebook 4,
# just restart the Space — it picks up the latest models.
# Live data comes from Kraken (no API key, no geo-restrictions).
# =============================================================================

import json
import os
import pickle
import warnings

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap

warnings.filterwarnings("ignore")

# ── Kraken API ────────────────────────────────────────────────────────────────

KRAKEN_OHLC   = "https://api.kraken.com/0/public/OHLC"
KRAKEN_TICKER = "https://api.kraken.com/0/public/Ticker"

# All symbols tracked by Producer + Notebooks
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


def _azure_container():
    """Return an Azure Blob container client using HuggingFace Space secrets."""
    from azure.storage.blob import BlobServiceClient
    conn_str       = os.environ["AZURE_STORAGE_CONN_STR"]
    container_name = os.environ["AZURE_CONTAINER"]
    client         = BlobServiceClient.from_connection_string(conn_str)
    return client.get_container_client(container_name)


# ── Model Loading ─────────────────────────────────────────────────────────────

def _load_models():
    container = _azure_container()
    clf    = pickle.loads(container.download_blob("models/clf.pkl").readall())
    reg    = pickle.loads(container.download_blob("models/reg.pkl").readall())
    scaler = pickle.loads(container.download_blob("models/scaler.pkl").readall())
    feats  = json.loads(container.download_blob("models/feature_names.json").readall())
    try:
        metrics = json.loads(container.download_blob("models/metrics.json").readall())
    except Exception:
        metrics = {}
    lsh_pairs = None
    try:
        data = json.loads(container.download_blob("notebook_data/lsh_similar_pairs.json").readall())
        lsh_pairs = pd.DataFrame(data)
    except Exception:
        pass
    return clf, reg, scaler, feats, metrics, lsh_pairs


try:
    clf, reg, scaler, FEATURE_NAMES, METRICS, LSH_PAIRS = _load_models()
    MODELS_OK = True
    print(f"✓ Models loaded | {len(FEATURE_NAMES)} features | "
          f"AUC={METRICS.get('classifier',{}).get('roc_auc','?')}")
except Exception as e:
    MODELS_OK = False
    clf = reg = scaler = FEATURE_NAMES = METRICS = LSH_PAIRS = None
    FEATURE_NAMES = []
    METRICS = {}
    print(f"[WARN] Models not loaded: {e}")


# ── Kraken Data Fetching ──────────────────────────────────────────────────────

def fetch_kraken_candles(pair: str, limit: int = 60) -> pd.DataFrame:
    """Fetch the last `limit` hourly candles from Kraken for one pair."""
    r = requests.get(KRAKEN_OHLC, params={"pair": pair, "interval": 60}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")
    result  = data.get("result", {})
    candles = next((v for k, v in result.items() if k != "last"), [])
    candles = candles[-limit:]
    df = pd.DataFrame(candles, columns=["open_time","open","high","low","close","vwap","volume","trade_count"])
    for col in ["open","high","low","close","vwap","volume"]:
        df[col] = df[col].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df["open_time"]   = df["open_time"].astype(int) * 1000   # to ms
    df["quote_volume"] = df["volume"] * df["vwap"]
    return df


def fetch_kraken_ticker(pair: str) -> dict:
    """Fetch 24h stats for one pair from Kraken."""
    try:
        r = requests.get(KRAKEN_TICKER, params={"pair": pair}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            return {}
        result = data.get("result", {})
        return next(iter(result.values()), {})
    except Exception:
        return {}


# ── Feature Engineering (mirrors Notebook 4 exactly) ─────────────────────────

def compute_features(df: pd.DataFrame) -> dict:
    """Compute all model features for the latest candle in df."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    lc = float(c.iloc[-1]); lo = float(o.iloc[-1])
    lh = float(h.iloc[-1]); ll = float(l.iloc[-1])
    lv = float(v.iloc[-1])

    body  = abs(lc - lo)
    rng   = lh - ll
    u_wick = lh - max(lc, lo)
    l_wick = min(lc, lo) - ll

    def sma(n): return float(c.tail(n).mean()) if len(c) >= n else float(c.mean())
    def pct(n): return float((c.iloc[-1] - c.iloc[-(n+1)]) / c.iloc[-(n+1)] * 100) if len(c) > n else 0.0

    log_ret = np.log(c / c.shift(1).replace(0, np.nan))

    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=2).std().fillna(0)
    bb_up = sma20 + 2 * std20
    bb_lo = sma20 - 2 * std20
    bb_w  = float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / sma20.iloc[-1]) if sma20.iloc[-1] else 0
    bb_p  = float((lc - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1])) if (bb_up.iloc[-1] - bb_lo.iloc[-1]) else 0.5

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = float(gain.iloc[-1]) / float(loss.iloc[-1]) if float(loss.iloc[-1]) else 0
    rsi   = 100 - (100 / (1 + rs))

    ema12    = c.ewm(span=12, adjust=False).mean()
    ema26    = c.ewm(span=26, adjust=False).mean()
    macd     = float((ema12 - ema26).iloc[-1])
    macd_sig = float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])

    vol_sma24  = float(v.rolling(24, min_periods=1).mean().iloc[-1])
    vol_ratio  = lv / vol_sma24 if vol_sma24 > 0 else 1.0
    vol_change = pct(1)

    obv     = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_max = float(obv.abs().max())
    obv_norm = float(obv.iloc[-1]) / obv_max if obv_max > 0 else 0

    return {
        "candle_body":     body,
        "candle_range":    rng,
        "upper_wick":      u_wick,
        "lower_wick":      l_wick,
        "body_ratio":      body / rng if rng > 0 else 0,
        "is_bullish":      int(lc > lo),
        "pct_change_1h":   pct(1),
        "pct_change_3h":   pct(3),
        "pct_change_6h":   pct(6),
        "pct_change_24h":  pct(24),
        "price_vs_sma_6":  (lc - sma(6))  / sma(6)  * 100 if sma(6)  > 0 else 0,
        "price_vs_sma_12": (lc - sma(12)) / sma(12) * 100 if sma(12) > 0 else 0,
        "price_vs_sma_24": (lc - sma(24)) / sma(24) * 100 if sma(24) > 0 else 0,
        "price_vs_sma_48": (lc - sma(48)) / sma(48) * 100 if len(c) >= 48 and sma(48) > 0 else 0,
        "macd":            macd,
        "macd_signal":     macd_sig,
        "macd_hist":       macd - macd_sig,
        "volatility_6h":   float(log_ret.rolling(6,  min_periods=2).std().iloc[-1]) if len(c) >= 6  else 0,
        "volatility_24h":  float(log_ret.rolling(24, min_periods=2).std().iloc[-1]) if len(c) >= 24 else 0,
        "bb_width":        bb_w,
        "bb_pct":          bb_p,
        "rsi":             rsi,
        "vol_change":      vol_change,
        "vol_ratio":       vol_ratio,
        "obv_norm":        obv_norm,
        "liquidity_ratio": 0,   # not available from Kraken
        "mcap_log":        0,   # not available from Kraken
    }


# ── Tab 1: Predict ────────────────────────────────────────────────────────────

def predict(pair: str):
    if not MODELS_OK:
        return "⚠️ Models not loaded. Check AZURE_STORAGE_CONN_STR and AZURE_CONTAINER secrets.", None, None

    try:
        df      = fetch_kraken_candles(pair, limit=60)
        feats   = compute_features(df)
        x_row   = np.array([[feats.get(f, 0) for f in FEATURE_NAMES]])
        x_scaled = scaler.transform(x_row)

        direction  = "📈 UP"   if clf.predict(x_scaled)[0] == 1 else "📉 DOWN"
        confidence = float(clf.predict_proba(x_scaled)[0].max()) * 100
        pct_change = float(reg.predict(x_scaled)[0])
        price      = float(df["close"].iloc[-1])

        base = SYMBOLS.get(pair, pair)
        result = f"""
### {base} ({pair}) — Prediction

| | |
|---|---|
| **Current Price** | ${price:,.4f} |
| **Predicted Direction** | {direction} |
| **Confidence** | {confidence:.1f}% |
| **Predicted % Change** | {pct_change:+.3f}% |
| **RSI (14)** | {feats['rsi']:.1f} |
| **Volatility 24h** | {feats['volatility_24h']:.4f} |
| **MACD** | {feats['macd']:.4f} |
| **BB Position** | {feats['bb_pct']:.2f} |

*Model: Random Forest | Test ROC-AUC: {METRICS.get('classifier',{}).get('roc_auc', '–')}*
"""

        # Candlestick chart
        closes = df["close"].values
        opens  = df["open"].values
        times  = pd.to_datetime(df["open_time"], unit="ms")

        fig_price, ax = plt.subplots(figsize=(10, 3.5))
        colors_candle = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(closes, opens)]
        ax.bar(range(len(closes)), closes - opens, bottom=opens,
               color=colors_candle, width=0.6, alpha=0.85)
        ax.plot(range(len(closes)), closes, color="#1565C0", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{base} — Last {len(closes)} Hourly Candles", fontsize=11)
        ax.set_ylabel("Price (USD)")
        tick_step = max(1, len(closes) // 6)
        ax.set_xticks(range(0, len(closes), tick_step))
        ax.set_xticklabels([times.iloc[i].strftime("%d %b %Hh")
                            for i in range(0, len(closes), tick_step)],
                           rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.2)
        fig_price.tight_layout()

        # SHAP waterfall
        fig_shap = None
        try:
            explainer = shap.TreeExplainer(clf)
            shap_exp  = explainer(x_scaled)
            fig_shap, _ = plt.subplots(figsize=(10, 6))
            exp = shap_exp[:, :, 1] if shap_exp.values.ndim == 3 else shap_exp
            shap.plots.waterfall(exp[0], max_display=15, show=False)
            plt.title(f"Why {base} is predicted {direction}", fontsize=11)
            plt.tight_layout()
        except Exception as e:
            print(f"[SHAP] {e}")

        return result, fig_price, fig_shap

    except Exception as e:
        return f"**Error:** {e}", None, None


# ── Tab 2: Market Overview ────────────────────────────────────────────────────

def market_overview():
    """Fetch live 24h stats for all tracked pairs from Kraken."""
    rows = []
    pairs_str = ",".join(SYMBOLS.keys())
    try:
        r    = requests.get(KRAKEN_TICKER, params={"pair": pairs_str}, timeout=15)
        data = r.json().get("result", {})
    except Exception:
        data = {}

    for pair, base in SYMBOLS.items():
        try:
            t          = data.get(pair) or data.get(f"X{pair}") or {}
            last_price = float(t.get("c", [0])[0])
            open_price = float(t.get("o", last_price))
            vol_24h    = float(t.get("v", [0, 0])[1])
            high_24h   = float(t.get("h", [0, 0])[1])
            low_24h    = float(t.get("l", [0, 0])[1])
            chg        = (last_price - open_price) / open_price * 100 if open_price else 0
            arrow      = "🟢" if chg >= 0 else "🔴"
            rows.append({
                "Pair":       pair,
                "Asset":      base,
                "Price ($)":  f"{last_price:,.4f}",
                "24h Change": f"{arrow} {chg:+.2f}%",
                "24h High":   f"{high_24h:,.4f}",
                "24h Low":    f"{low_24h:,.4f}",
                "24h Volume": f"{vol_24h:,.0f}",
            })
        except Exception:
            rows.append({"Pair": pair, "Asset": base, "Price ($)": "–",
                         "24h Change": "–", "24h High": "–",
                         "24h Low": "–", "24h Volume": "–"})

    return pd.DataFrame(rows)


# ── Tab 3: Similar Coins ──────────────────────────────────────────────────────

def find_similar(pair: str, top_n: int = 8) -> str:
    if LSH_PAIRS is None or len(LSH_PAIRS) == 0:
        return ("LSH similarity data not available yet. "
                "Run Notebook 3 first — it uploads lsh_similar_pairs.json to Azure Blob automatically.")
    mask = (LSH_PAIRS["symbol_a"] == pair) | (LSH_PAIRS["symbol_b"] == pair)
    df   = LSH_PAIRS[mask].copy()
    if df.empty:
        return f"No similar pairs found for {pair}."
    df["other"] = df.apply(
        lambda r: r["symbol_b"] if r["symbol_a"] == pair else r["symbol_a"], axis=1
    )
    df = df.sort_values("distance").head(top_n)
    base  = SYMBOLS.get(pair, pair)
    lines = [f"### Coins most similar to {base}", ""]
    for _, row in df.iterrows():
        other_base = SYMBOLS.get(row["other"], row["other"])
        bar = "█" * int(row["similarity"] * 20)
        lines.append(f"**{other_base}** (`{row['other']}`)  "
                     f"similarity={row['similarity']:.3f}  {bar}")
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

pair_list = list(SYMBOLS.keys())

with gr.Blocks(title="🔮 Crypto Predictor", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔮 Crypto Price Direction Predictor
    **Predicts next-hour price direction (UP/DOWN) using a Random Forest trained on Kraken OHLCV data.**
    Live data from Kraken • No API key needed • SHAP explanations for every prediction
    """)

    with gr.Tab("🎯 Predict"):
        with gr.Row():
            sym_dd   = gr.Dropdown(choices=pair_list, value="XBTUSD", label="Select Pair", scale=3)
            pred_btn = gr.Button("🔮 Predict", variant="primary", scale=1)
        result_md = gr.Markdown()
        with gr.Row():
            price_plot = gr.Plot(label="Price History (Candlestick)")
            shap_plot  = gr.Plot(label="SHAP — Why this prediction?")
        pred_btn.click(fn=predict, inputs=[sym_dd], outputs=[result_md, price_plot, shap_plot])

    with gr.Tab("📊 Market Overview"):
        refresh_btn  = gr.Button("🔄 Refresh", variant="secondary")
        overview_tbl = gr.Dataframe(label="Live Market Data (Kraken)")
        refresh_btn.click(fn=market_overview, outputs=overview_tbl)
        demo.load(fn=market_overview, outputs=overview_tbl)

    with gr.Tab("🔗 Similar Coins (LSH)"):
        gr.Markdown("Coins with similar market behaviour, found via Locality Sensitive Hashing (Notebook 3).")
        with gr.Row():
            lsh_sym = gr.Dropdown(choices=pair_list, value="XBTUSD", label="Select Pair", scale=3)
            lsh_btn = gr.Button("Find Similar", scale=1)
        lsh_result = gr.Markdown()
        lsh_btn.click(fn=find_similar, inputs=[lsh_sym], outputs=lsh_result)

    with gr.Tab("ℹ️ About"):
        auc_str = str(METRICS.get("classifier", {}).get("roc_auc", "–"))
        mae_str = str(METRICS.get("regressor",  {}).get("mae",     "–"))
        n_feat  = str(len(FEATURE_NAMES))
        n_train = str(METRICS.get("training_samples", "–"))
        syms    = METRICS.get("symbols", list(SYMBOLS.keys()))
        gr.Markdown(f"""
## Architecture

```
Kraken API (free, no key, no geo-restrictions)
      │
      ├── Oracle Cloud VM (24/7 free)
      │     Producer.py → Azure Service Bus → Consumer.py → Azure Blob
      │
      └── OR: Kaggle Notebook 1 (on-demand fallback)
                    │
            Azure Blob Storage (5 GB free for 12 months)
                    │
      ┌─────────────┼─────────────┐
      │             │             │
  Notebook 2    Notebook 3    Notebook 4
  MapReduce      FAISS LSH    ML + SHAP
      │             │             │
      └─────────────┴─────────────┘
                    │ models/ + notebook_data/ in Azure Blob
                    │
          HuggingFace Spaces (this app)
          Live Kraken data → predictions
```

## Model Stats
| Metric | Value |
|--------|-------|
| **Classifier ROC-AUC (test)** | {auc_str} |
| **Regressor MAE (test)** | {mae_str}% |
| **Features** | {n_feat} |
| **Training samples** | {n_train} |
| **Symbols tracked** | {len(syms)} |

## Features ({n_feat} total)
Candle anatomy (body, wicks, range, body ratio), price momentum (1h/3h/6h/24h %),
SMAs (6/12/24/48h deviations), MACD (12/26/9), RSI-14, Bollinger Bands (width + position),
volatility (6h/24h log-return std), volume ratio, OBV (normalised).

## Free Stack
| Component | Service |
|-----------|---------|
| Market data | Kraken API |
| Streaming broker | Azure Service Bus (13M msgs/month free for 12 months) |
| 24/7 VM | Oracle Cloud Always Free (2 VMs forever) |
| Storage | Azure Blob Storage (5 GB free for 12 months) |
| ML compute | Kaggle Notebooks (30h/week free) |
| App hosting | HuggingFace Spaces (Gradio, always-on, free) |
        """)


if __name__ == "__main__":
    demo.launch()
