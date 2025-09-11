import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta, date
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

ROLLING_WINDOW = 5  # default

# -------------------
# Year Selection Dropdown
# -------------------
year_range = list(range(2022, 2026))
selected_year = st.sidebar.selectbox("Select Year", options=year_range)
start_date = date(selected_year, 1, 1)
end_date = date(selected_year, 12, 31)
st.sidebar.write(f"Data Range: {start_date} to {end_date}")

# -------------------
# Helper: Convert date range to Yahooquery period
# -------------------
def date_range_to_period(start_date, end_date):
    delta_days = (end_date - start_date).days
    if delta_days < 1:
        delta_days = 1
    return f"{delta_days}d"

# -------------------
# Fetch data and calculate RVol
# -------------------
def fetch_rvol_data(symbol, start_date, end_date, window=ROLLING_WINDOW):
    try:
        period = date_range_to_period(start_date, end_date)
        t = Ticker(symbol, timeout=60)
        hist = t.history(period=period, interval="1h")

        # Handle dict response
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

            # Convert to GMT+3 for display
            hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
            hist.set_index("datetime_gmt3", inplace=True)

            # Ensure numeric and remove NaN from rolling
            hist["avg_volume"] = hist["volume"].rolling(window).mean()
            hist["rvol"] = hist["volume"] / hist["avg_volume"]
            hist = hist.dropna(subset=["rvol", "close"])

            # Ensure index is datetime
            hist.index = pd.to_datetime(hist.index)

            return hist[["open", "high", "low", "close", "volume", "rvol"]]

        st.warning(f"No data returned for {symbol}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

# -------------------
# Threaded fetching (process then cache)
# -------------------
def fetch_process(symbol):
    df = fetch_rvol_data(symbol, start_date, end_date, ROLLING_WINDOW)
    return symbol, df

def fetch_all_tickers(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_process, symbol): symbol for symbol in tickers}
        for future in as_completed(futures):
            symbol, df = future.result()
            results[symbol] = df
    return results

@st.cache_data(show_spinner=True)
def get_all_rvol_data(selected_assets, start_date, end_date, window, threshold):
    symbols = [TICKER_MAP[a] for a in selected_assets]
    return fetch_all_tickers(symbols)

# -------------------
# Gap-Up Detection
# -------------------
def detect_gap_up(df, session_hours, threshold=2.0):
    if df is None or df.empty:
        return None, None, None

    df = df.copy()
    df["date"] = df.index.date
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
# Run Backtest
# -------------------
if st.sidebar.button("Run Backtest"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    all_results = []
    total_assets = len(selected_assets)

    for i, symbol_name in enumerate(selected_assets):
        symbol = TICKER_MAP[symbol_name]
        status_text.text(f"Processing {symbol_name} ({i+1}/{total_assets})...")

        trades, symbol = backtest_asset(symbol, session_hours, threshold, holding_period)

        progress_bar.progress((i + 1) / total_assets)

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

        # Summary metrics
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

        # Trades table
        st.subheader("Trade Details")
        st.dataframe(df_results.sort_values("entry_date", ascending=False))

        # Return distribution
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
        fig2.update_layout(title="Performance by Asset", yaxis_title="Value")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No trades generated in the selected period. Adjust parameters or select different assets.")
else:
    st.info("Configure the backtest parameters in the sidebar and click 'Run Backtest' to start.")