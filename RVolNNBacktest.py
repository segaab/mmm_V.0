import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide")
st.title("📊 RVol Gap-Up Backtester")

# -------------------
# Asset Universe
# -------------------
TICKER_MAP = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Corn": "ZC=F",
    "Wheat": "ZW=F",
    "Soybeans": "ZS=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

# -------------------
# Data Fetching
# -------------------
@st.cache_data(show_spinner=False)
def get_data(ticker_symbol, start_date, end_date):
    ticker = Ticker(ticker_symbol)
    df = ticker.history(
        start=start_date,
        end=end_date,
        interval="1h"
    ).reset_index()

    if df.empty:
        return None

    df.rename(columns={"symbol": "ticker"}, inplace=True)
    df["datetime_gmt3"] = pd.to_datetime(df["date"])
    df.set_index("datetime_gmt3", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]

    return df


# -------------------
# RVol Calculation
# -------------------
def calculate_rvol(df, window=5):
    df = df.copy()
    df["avg_vol"] = df["volume"].rolling(window=window).mean()
    df["rvol"] = df["volume"] / df["avg_vol"]
    return df


# -------------------
# Gap-Up Detection
# -------------------
def detect_gap_up(df, session_hours, threshold=2.0):
    if df is None or df.empty:
        return None, None, None
        
    df = df.copy()
    df = df.sort_index()  # Ensure chronological order
    
    # Extract date and hour information
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    
    # Get unique dates in descending order
    dates = sorted(df["date"].unique(), reverse=True)
    if len(dates) < 2:
        return None, None, None
    
    latest_day = dates[0]
    prev_day = dates[1]
    
    # Get RVol data for session hours on both days
    curr_open = df[(df["date"] == latest_day) & (df["hour"].isin(session_hours))]["rvol"]
    prev_open = df[(df["date"] == prev_day) & (df["hour"].isin(session_hours))]["rvol"]
    
    if curr_open.empty or prev_open.empty:
        return None, None, None
        
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    
    if prev_mean == 0 or pd.isna(prev_mean):
        return None, curr_mean, prev_mean
        
    gap_ratio = curr_mean / prev_mean
    
    return gap_ratio, curr_mean, prev_mean



# -------------------
# Backtest Logic
# -------------------
def backtest_asset(symbol, session_hours, start_date, end_date, window=5):
    df = get_data(symbol, start_date, end_date)
    if df is None or df.empty:
        return None, symbol

    df = calculate_rvol(df, window)

    gap_ratio, curr_mean, prev_mean = detect_gap_up(df, session_hours)

    if gap_ratio is None or np.isnan(gap_ratio):
        return None, symbol

    # Simple rule: if gap_ratio > 2, assume long, else no signal
    trades = []
    if gap_ratio > 2:
        entry_price = df["close"].iloc[-1]
        exit_price = df["close"].iloc[-1] * 1.01  # assume +1% move
        trades.append({"entry": entry_price, "exit": exit_price, "pnl": exit_price - entry_price})

    return trades, symbol


# -------------------
# Sidebar Settings
# -------------------
st.sidebar.header("Backtesting Settings")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31").date())

session_choice = st.sidebar.selectbox(
    "Select Entry Session",
    ["Asian (3–4 GMT)", "London (10–11 GMT)", "New York (16–17 GMT)"]
)

session_map = {
    "Asian (3–4 GMT)": [3, 4],
    "London (10–11 GMT)": [10, 11],
    "New York (16–17 GMT)": [16, 17],
}
session_hours = session_map[session_choice]

rolling_window = st.sidebar.number_input("RVol Rolling Window (hours)", min_value=3, max_value=50, value=5)


# -------------------
# Multi-Asset Backtesting
# -------------------
selected_assets = st.sidebar.multiselect(
    "Select Assets", options=list(TICKER_MAP.keys()), default=["Gold", "EUR/USD", "Crude Oil"]
)

results = []
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            backtest_asset, TICKER_MAP[symbol], session_hours, str(start_date), str(end_date), rolling_window
        )
        for symbol in selected_assets
    ]
    for fut in futures:
        trades, symbol = fut.result()
        if trades:
            for t in trades:
                t["asset"] = symbol
                results.append(t)

# -------------------
# Display Results
# -------------------
if results:
    df_results = pd.DataFrame(results)
    st.subheader("Backtest Results")
    st.dataframe(df_results)

    df_results["return_pct"] = df_results["pnl"] / df_results["entry"] * 100

    win_rate = (df_results["pnl"] > 0).mean() * 100
    avg_rr = df_results["return_pct"].mean()

    st.metric("Win Rate (%)", f"{win_rate:.2f}")
    st.metric("Avg Return (%)", f"{avg_rr:.2f}")

    # Distribution of returns
    st.subheader("Distribution of Returns")
    fig = px.histogram(df_results, x="return_pct", nbins=20, title="Return Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No trades generated in the selected period.")