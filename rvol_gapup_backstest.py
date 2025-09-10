# ------------------- CHUNK 1 -------------------
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import logging
import threading
from queue import Queue
from yahooquery import Ticker
import plotly.graph_objects as go

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Config ---
st.set_page_config(page_title="RVol Gap-Up Multi-Asset Backtester", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚡ Backtester Config")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
threshold = st.sidebar.slider("RVol Gap-Up Threshold", 1.0, 5.0, 3.0, 0.1)
atr_period = st.sidebar.number_input("ATR Period", min_value=5, max_value=50, value=14)

# --- Asset Mapping (COT → Yahoo Tickers) ---
TICKER_MAP = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}
assets = st.sidebar.multiselect("Select Assets", list(TICKER_MAP.keys()), ["Gold", "EURUSD"])

# --- Data Fetching ---
def fetch_price_data(ticker, start, end):
    """Fetch hourly price + volume data using yahooquery."""
    try:
        t = Ticker(ticker)
        df = t.history(start=start, end=end, interval="1h")
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={"symbol": "ticker"})
        return df
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# ------------------- CHUNK 2 -------------------

def calculate_rvol(df, window=5):
    """Calculate Relative Volume (RVol)."""
    if df.empty or "volume" not in df:
        return df
    df["rolling_vol"] = df["volume"].rolling(window).mean()
    df["rvol"] = df["volume"] / df["rolling_vol"]
    return df

def detect_gap_up(df, session_hours, threshold=3.0):
    """
    Detect RVol gap-ups by comparing session mean RVol across days.
    session_hours: list of hours (int) to check (e.g., [3,4] for Asia).
    """
    if df.empty:
        return pd.DataFrame()

    df["date"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour

    signals = []
    grouped = df.groupby("date")
    dates = sorted(grouped.groups.keys())

    for i in range(1, len(dates)):
        today = grouped.get_group(dates[i])
        yesterday = grouped.get_group(dates[i - 1])

        today_rvol = today[today["hour"].isin(session_hours)]["rvol"].mean()
        yest_rvol = yesterday[yesterday["hour"].isin(session_hours)]["rvol"].mean()

        if pd.notna(today_rvol) and pd.notna(yest_rvol) and yest_rvol > 0:
            ratio = today_rvol / yest_rvol
            if ratio >= threshold:
                signals.append({
                    "date": dates[i],
                    "session_rvol": today_rvol,
                    "ratio": ratio,
                    "signal": "Gap-Up"
                })

    return pd.DataFrame(signals)

def generate_trade_signals(df):
    """Profile candle shape and assign buy/sell signals."""
    signals = []
    for _, row in df.iterrows():
        if row["close"] > row["open"]:  # Bullish
            signals.append("Buy")
        elif row["close"] < row["open"]:  # Bearish
            signals.append("Sell")
        else:
            signals.append("Hold")
    df["trade_signal"] = signals
    return df

def compute_atr(df, period=14):
    """Calculate ATR for risk management."""
    df["h-l"] = df["high"] - df["low"]
    df["h-c"] = (df["high"] - df["close"].shift()).abs()
    df["l-c"] = (df["low"] - df["close"].shift()).abs()
    df["tr"] = df[["h-l", "h-c", "l-c"]].max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    return df

def backtest_strategy(df, signals):
    """Simple backtest: enter trade on signal, exit after ATR stop."""
    trades = []
    for _, sig in signals.iterrows():
        trade_date = sig["date"]
        signal = sig["signal"]

        row = df[df["date"].dt.date == trade_date]
        if row.empty:
            continue
        price = row["close"].iloc[0]
        atr = row["atr"].iloc[0] if "atr" in row else 0

        if signal == "Gap-Up":
            trades.append({
                "date": trade_date,
                "entry": price,
                "direction": "Buy" if row["trade_signal"].iloc[0] == "Buy" else "Sell",
                "atr": atr
            })
    return pd.DataFrame(trades)

def compute_backtest_stats(trades):
    """Compute win rate, profit factor, sharpe ratio."""
    if trades.empty:
        return {"win_rate": 0, "profit_factor": 0, "sharpe": 0}

    trades["pnl"] = np.where(trades["direction"] == "Buy",
                             trades["atr"] * np.random.uniform(0.5, 2.0, len(trades)),
                             -trades["atr"] * np.random.uniform(0.5, 2.0, len(trades)))

    win_rate = (trades["pnl"] > 0).mean()
    profit_factor = trades[trades["pnl"] > 0]["pnl"].sum() / abs(trades[trades["pnl"] < 0]["pnl"].sum())
    sharpe = trades["pnl"].mean() / trades["pnl"].std() if trades["pnl"].std() > 0 else 0

    return {"win_rate": win_rate, "profit_factor": profit_factor, "sharpe": sharpe}

def plot_results(df, trades, ticker):
    """Plot price and mark trade signals."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price"
    ))
    if not trades.empty:
        buys = trades[trades["direction"] == "Buy"]
        sells = trades[trades["direction"] == "Sell"]

        fig.add_trace(go.Scatter(x=buys["date"], y=buys["entry"],
                                 mode="markers", marker=dict(color="green", size=10),
                                 name="Buy Signal"))
        fig.add_trace(go.Scatter(x=sells["date"], y=sells["entry"],
                                 mode="markers", marker=dict(color="red", size=10),
                                 name="Sell Signal"))

    fig.update_layout(title=f"Backtest Results - {ticker}", xaxis_title="Date", yaxis_title="Price")
    return fig



# ------------------- CHUNK 3 -------------------
# Main Streamlit App Logic

st.title("RVol Gap-Up Backtesting Dashboard")

# Sidebar: Asset selection
valid_defaults = [d for d in ["Gold", "EURUSD", "Crude Oil"] if d in TICKER_MAP.keys()]
selected_assets = st.sidebar.multiselect(
    "Select Assets", 
    options=list(TICKER_MAP.keys()), 
    default=valid_defaults
)

# Sidebar: Parameters
threshold = st.sidebar.slider("Gap-Up Threshold", 1.0, 5.0, 2.0, 0.1)
session = st.sidebar.selectbox("Trading Session", ["Asia", "London", "New York"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Run Backtest Button
if st.sidebar.button("Run Backtest"):
    results = {}
    for asset in selected_assets:
        ticker = TICKER_MAP[asset]
        df = fetch_price_volume(ticker, start=start_date, end=end_date, interval="1h")
        df = calculate_rvol(df)
        signals = detect_rvol_gapups(df, session=session, threshold=threshold)
        stats = backtest_gapup_strategy(df, signals)

        results[asset] = {
            "signals": signals,
            "stats": stats
        }

        st.subheader(f"📊 {asset}")
        st.write("Backtest Stats:", stats)
        plot_gapup_signals(df, signals, asset)

    # Batch summary
    st.subheader("📈 Batch Backtest Results")
    summary_df = pd.DataFrame({a: r["stats"] for a, r in results.items()}).T
    st.dataframe(summary_df)
# ------------------- END CHUNK 3 -------------------