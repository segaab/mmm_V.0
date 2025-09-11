import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
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

# -------------------
# Data Fetching + RVol Calculation
# -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol, start_date, end_date, window=5):
    """Fetch hourly data, trim to date range, and compute RVol"""
    try:
        t = Ticker(symbol, timeout=60)
        hist = t.history(start=start_date, end=end_date, interval="1h")

        if isinstance(hist, pd.DataFrame) and not hist.empty:
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.reset_index()

            hist = hist.rename(columns={"symbol": "ticker"})
            hist = hist.dropna(subset=["volume", "date"])
            hist = hist[hist["volume"] > 0]

            # Parse datetime
            hist["datetime"] = pd.to_datetime(hist["date"], errors="coerce", utc=True)
            hist = hist.dropna(subset=["datetime"])
            hist = hist.sort_values("datetime")

            # Convert to GMT+3
            hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
            hist.set_index("datetime_gmt3", inplace=True)

            # Calculate avg_volume and rvol
            hist["avg_volume"] = hist["volume"].rolling(window).mean()
            hist["rvol"] = hist["volume"] / hist["avg_volume"]

            # Keep only needed columns
            return hist[["open", "high", "low", "close", "volume", "rvol"]].dropna()

        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

# -------------------
# Gap-Up Detection
# -------------------
def detect_gap_up(df, session_hours, threshold=2.0):
    """Detect gap-up in relative volume between consecutive days for specified session hours"""
    if df is None or df.empty:
        return None, None, None

    df = df.copy()
    df["date"] = df.index.date
    df["hour"] = df.index.hour

    dates = sorted(df["date"].unique(), reverse=False)
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
def backtest_asset(symbol, session_hours, start_date, end_date, threshold=2.0, window=5, holding_period=5):
    """Backtest a single asset for RVol gap-up strategy"""
    df = fetch_rvol_data(symbol, start_date, end_date, window)
    if df is None or df.empty:
        return None, symbol

    trades = []
    dates = sorted(df.index.date.unique())

    for i in range(1, len(dates) - holding_period):
        temp_df = df[df.index.date <= dates[i]].copy()
        gap_ratio, curr_mean, prev_mean = detect_gap_up(temp_df, session_hours, threshold)

        if gap_ratio is not None and gap_ratio >= threshold:
            entry_date = dates[i]
            entry_price = df[df.index.date == entry_date]["close"].iloc[-1]

            exit_date = dates[i + holding_period]
            exit_df = df[df.index.date == exit_date]

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

# -------------------
# Sidebar Settings
# -------------------
st.sidebar.header("Backtesting Settings")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31").date())

session_choice = st.sidebar.selectbox(
    "Select Entry Session",
    ["Asian (3-4 GMT+3)", "London (10-11 GMT+3)", "New York (16-17 GMT+3)"]
)

session_map = {
    "Asian (3-4 GMT+3)": [3, 4],
    "London (10-11 GMT+3)": [10, 11],
    "New York (16-17 GMT+3)": [16, 17],
}
session_hours = session_map[session_choice]

rolling_window = st.sidebar.number_input("RVol Rolling Window (hours)", min_value=3, max_value=50, value=5)
threshold = st.sidebar.number_input("Gap-Up Threshold", min_value=1.1, max_value=5.0, value=2.0, step=0.1)
holding_period = st.sidebar.number_input("Holding Period (days)", min_value=1, max_value=10, value=5)

selected_assets = st.sidebar.multiselect(
    "Select Assets", options=list(TICKER_MAP.keys()), default=["Gold", "EUR/USD", "Crude Oil"]
)

# -------------------
# Run Backtest
# -------------------
if st.sidebar.button("Run Backtest"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_results = []
    total_assets = len(selected_assets)

    for i, symbol_name in enumerate(selected_assets):
        symbol = TICKER_MAP[symbol_name]
        status_text.text(f"Fetching data for {symbol_name}...")

        trades, symbol = backtest_asset(
            symbol, session_hours, str(start_date), str(end_date),
            threshold, rolling_window, holding_period
        )

        progress_bar.progress((i + 1) / total_assets)
        status_text.text(f"Processed {symbol} ({i+1}/{total_assets})")

        if trades:
            for t in trades:
                t["asset"] = symbol_name
                all_results.append(t)

    progress_bar.empty()
    status_text.empty()

    # -------------------
    # Display Results
    # -------------------
    if all_results:
        df_results = pd.DataFrame(all_results)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            win_rate = (df_results["pnl"] > 0).mean() * 100
            st.metric("Win Rate", f"{win_rate:.2f}%")

        with col2:
            avg_return = df_results["return_pct"].mean()
            st.metric("Avg Return", f"{avg_return:.2f}%")

        with col3:
            total_trades = len(df_results)
            st.metric("Total Trades", total_trades)

        with col4:
            if len(df_results) > 1:
                sharpe = df_results["return_pct"].mean() / df_results["return_pct"].std()
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.subheader("Trade Details")
        st.dataframe(df_results.sort_values("entry_date", ascending=False))

        st.subheader("Return Distribution")
        fig = px.histogram(
            df_results, x="return_pct", nbins=20,
            title="Return Distribution (%)",
            color_discrete_sequence=["#3366CC"]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

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
        fig2.update_layout(title="Performance by Asset", yaxis_title="Value")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No trades generated in the selected period. Try adjusting parameters or selecting different assets.")
else:
    st.info("Configure the backtest parameters in the sidebar and click 'Run Backtest' to start.")