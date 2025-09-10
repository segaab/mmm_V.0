import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import timedelta, datetime
import plotly.graph_objs as go

# -------------------------------
# Ticker map (assets only)
# -------------------------------
TICKER_MAP = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC-USD",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH-USD",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "DX-Y.NYB",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6N=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "^DJI",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "COPPER - COMMODITY EXCHANGE INC.":"HG=F"
}

TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}
symbols = list(TICKER_MAP.values())

# -------------------------------
# Sidebar - Backtesting Settings
# -------------------------------
st.sidebar.header("Backtesting Settings")

start_date = st.sidebar.date_input(
    "Start Date", pd.to_datetime("2023-01-01").date(), key="start_date"
)
end_date = st.sidebar.date_input(
    "End Date", pd.to_datetime("2025-01-01").date(), key="end_date"
)

entry_session = st.sidebar.selectbox(
    "Select Entry Session:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"],
    key="entry_session"
)

rolling_window = st.sidebar.number_input(
    "RVol Rolling Window (hours)", min_value=1, max_value=500, value=120, step=1,
    key="rolling_window"
)

st.title("RVol Gap-Up Backtester")

# -------------------------------
# Determine market open hours based on session
# -------------------------------
if entry_session.startswith("London"):
    open_hours = [10, 11]
elif entry_session.startswith("NY"):
    open_hours = [16, 17]
elif entry_session.startswith("Asian"):
    open_hours = [3, 4]
else:
    open_hours = [16, 17]

# -------------------------------
# Functions
# -------------------------------
def detect_gap_up(df, open_hours, threshold=1.5):
    if df.empty:
        return False, None, None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    if df.empty:
        return False, None, None
    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - pd.Timedelta(days=1)
    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    if curr_open.empty or prev_open.empty:
        return False, None, None
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    if prev_mean == 0 or pd.isna(prev_mean):
        return False, curr_mean, prev_mean
    gap_ratio = curr_mean / prev_mean
    return gap_ratio >= threshold, curr_mean, prev_mean

@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol):
    t = Ticker(symbol, timeout=60)
    hist = t.history(period="730d", interval="1h")
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
        hist["datetime_gmt3"] = hist["datetime_gmt3"].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        # Calculate rolling average volume and rvol
        hist["avg_volume"] = hist["volume"].rolling(rolling_window).mean()
        hist["rvol"] = hist["volume"] / hist["avg_volume"]
        # Filter by backtesting date range
        hist = hist[(hist["datetime_gmt3_dt"] := pd.to_datetime(hist["datetime_gmt3"], errors="coerce")) >= pd.to_datetime(start_date)]
        hist = hist[hist["datetime_gmt3_dt"] <= pd.to_datetime(end_date)]
        return hist
    else:
        return pd.DataFrame()

# -------------------------------
# Backtesting and RVol Gap-Up Analysis
# -------------------------------
st.header("RVol Gap-Up Backtesting Results")

distribution_results = []

for symbol in asset_symbols:
    asset_name = TICKER_TO_NAME.get(symbol, symbol)
    df = fetch_rvol_data(symbol)
    if df.empty:
        st.warning(f"No data found for {asset_name} ({symbol}) in the selected date range.")
        continue

    # Gap-up detection per day
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour

    daily_results = []
    unique_dates = sorted(df["date_gmt3"].unique())
    for day in unique_dates[1:]:  # Skip the first day, no previous day to compare
        prev_day = day - pd.Timedelta(days=1)
        curr_open = df[(df["date_gmt3"] == day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
        prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
        if curr_open.empty or prev_open.empty:
            continue
        curr_mean = curr_open.mean()
        prev_mean = prev_open.mean()
        gap_ratio = curr_mean / prev_mean if prev_mean != 0 else 0
        daily_results.append(gap_ratio)

    if not daily_results:
        st.info(f"No valid gap-up days found for {asset_name} ({symbol}).")
        continue

    # Calculate distribution statistics
    daily_series = pd.Series(daily_results)
    percentile_values = [50, 70, 80, 90, 95]
    percentiles = daily_series.quantile([p/100 for p in percentile_values])

    # Display summary stats
    st.subheader(f"{asset_name} ({symbol}) Gap-Up Distribution")
    st.write(daily_series.describe())
    st.write("Selected Percentiles:")
    for p, val in zip(percentile_values, percentiles):
        st.write(f"{p}th percentile: {val:.2f}")

    # Plot histogram
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=daily_series, nbinsx=30, marker_color="blue"))
    fig.update_layout(
        title=f"Gap-Up RVol Distribution — {asset_name} ({symbol})",
        xaxis_title="Gap-Up Ratio",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    distribution_results.append({
        "asset": asset_name,
        "symbol": symbol,
        "daily_gap_ratios": daily_series
    })

st.success("Backtesting complete. Distribution stats and gap-up ratios displayed for all assets.")