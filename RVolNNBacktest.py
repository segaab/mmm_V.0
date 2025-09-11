import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
import time

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
    "Copper": "HG=F",
    "S&P 500": "ES=F",
    "Nasdaq": "NQ=F",
}

ROLLING_WINDOW = 5  # Default RVol window

# -------------------
# Year Selection Dropdown
# -------------------
year_range = list(range(2022, 2026))  # 2022–2025
selected_year = st.sidebar.selectbox("Select Year", options=year_range)

# Generate start_date and end_date for the selected year
start_date = date(selected_year, 1, 1)
end_date = date(selected_year, 12, 31)
st.sidebar.write(f"Data Range: {start_date} to {end_date}")

# -------------------
# Helper: Convert start/end to Yahooquery period
# -------------------
def date_range_to_period(start_date, end_date):
    delta_days = (end_date - start_date).days
    if delta_days < 1:
        delta_days = 1
    return f"{delta_days}d"

# -------------------
# Data Fetching + RVol Calculation
# -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol, start_date, end_date, window=ROLLING_WINDOW):
    """Fetch hourly data and compute RVol using Yahooquery period"""
    try:
        period = date_range_to_period(start_date, end_date)
        t = Ticker(symbol, timeout=60)
        hist = t.history(period=period, interval="1h")

        # Handle dictionary return
        if isinstance(hist, dict):
            key = list(hist.keys())[0]
            hist = hist[key]

        if isinstance(hist, pd.DataFrame) and not hist.empty:
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.reset_index()

            hist = hist.rename(columns={"symbol": "ticker"})
            hist = hist.dropna(subset=["volume", "date"])
            hist = hist[hist["volume"] > 0]

            hist["datetime"] = pd.to_datetime(hist["date"], errors="coerce", utc=True)
            hist = hist.dropna(subset=["datetime"])
            hist = hist.sort_values("datetime")

            # Convert to GMT+3
            hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
            hist.set_index("datetime_gmt3", inplace=True)

            # Calculate avg_volume and rvol
            hist["avg_volume"] = hist["volume"].rolling(window).mean()
            hist["rvol"] = hist["volume"] / hist["avg_volume"]

            return hist[["open", "high", "low", "close", "volume", "rvol"]].dropna()

        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

# -------------------
# Gap-Up Detection
# -------------------
def detect_gap_up(df, session_hours, threshold=2.0):
    if df is None or df.empty:
        return None, None, None

    df = df.copy()
    df["date"] = df.index.map(lambda x: x.date)  # Fixed AttributeError
    df["hour"] = df.index.hour

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return None, None, None

    results = []
    for i in range(1, len(dates)):
        current_day = dates[i]
        prev_day = dates[i - 1]

        curr_session = df[(df["date"] == current_day) & (df["hour"].isin(session_hours))]["rvol"]
        prev_session = df[(df["date"] == prev_day) & (df["hour"].isin(session_hours))]["rvol"]

        if not curr_session.empty and not prev_session.empty:
            curr_mean = curr_session.mean()
            prev_mean = prev_session.mean()

            if prev_mean > 0 and not pd.isna(prev_mean):
                gap_ratio = curr_mean / prev_mean
                results.append((current_day, gap_ratio, curr_mean, prev_mean))

    if not results:
        return None, None, None

    for date, ratio, curr, prev in reversed(results):
        if ratio >= threshold:
            return ratio, curr, prev

    date, ratio, curr, prev = results[-1]
    return ratio, curr, prev

# -------------------
# Backtesting Logic
# -------------------
def backtest_asset(symbol, session_hours, start_date, end_date, threshold=2.0, window=ROLLING_WINDOW, holding_period=5):
    df = fetch_rvol_data(symbol, start_date, end_date, window)
    if df is None or df.empty:
        return None, symbol

    trades = []
    dates = sorted(df.index.map(lambda x: x.date).unique())  # Fixed AttributeError

    for i in range(1, len(dates) - holding_period):
        temp_df = df[df.index.map(lambda x: x.date) <= dates[i]].copy()
        gap_ratio, curr_mean, prev_mean = detect_gap_up(temp_df, session_hours, threshold)

        if gap_ratio is not None and gap_ratio >= threshold:
            entry_date = dates[i]
            entry_price = df[df.index.map(lambda x: x.date) == entry_date]["close"].iloc[-1]

            exit_date = dates[i + holding_period]
            exit_df = df[df.index.map(lambda x: x.date) == exit_date]

            if not exit_df.empty:
                exit_price = exit_df["close"].iloc[-1]
                pnl = exit_price - entry_price
                pct_return = (exit_price / entry_price - 1) * 100

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": pct_return,
                    "gap_ratio": gap_ratio,
                    "rvol_curr": curr_mean,
                    "rvol_prev": prev_mean
                })

    return trades, symbol




st.subheader("Return Distribution")
        fig = px.histogram(
            df_results,
            x="return_pct",
            nbins=20,
            title="Return Distribution (%)",
            color_discrete_sequence=["#3366CC"]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        # Performance by asset
        st.subheader("Performance by Asset")
        asset_perf = df_results.groupby("asset").agg({
            "return_pct": ["mean", "count"],
            "pnl": lambda x: (x > 0).mean() * 100
        }).reset_index()
        asset_perf.columns = ["asset", "avg_return", "num_trades", "win_rate"]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=asset_perf["asset"],
            y=asset_perf["avg_return"],
            name="Avg Return (%)",
            marker_color="#3366CC"
        ))
        fig2.add_trace(go.Scatter(
            x=asset_perf["asset"],
            y=asset_perf["win_rate"],
            name="Win Rate (%)",
            mode="markers",
            marker=dict(size=12, color="#FF9900")
        ))
        fig2.update_layout(
            title="Performance by Asset",
            yaxis_title="Value",
            xaxis_title="Asset",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No trades generated in the selected period. Try adjusting parameters or selecting different assets.")
else:
    st.info("Configure the backtest parameters in the sidebar and click 'Run Backtest' to start.")