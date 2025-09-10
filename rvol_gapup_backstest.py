import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import timedelta, datetime
import plotly.graph_objs as go

# ------------------- ASSETS -------------------
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
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "^DJI",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "COPPER - COMMODITY EXCHANGE INC.":"HG=F"
}

asset_symbols = list(TICKER_MAP.values())
TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}

DAYS = 730  # Default 2 years rolling window
ROLLING_WINDOW = 120

# ------------------- STREAMLIT PAGE -------------------
st.title("RVol Gap-Up Backtester")

# Backtesting date range
st.sidebar.header("Backtesting Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.today().date()))
if start_date > end_date:
    st.sidebar.error("Start Date must be before End Date.")

# Select entry session
market_open_session = st.sidebar.selectbox(
    "Select Entry Session:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)
if market_open_session.startswith("London"):
    open_hours = [10, 11]
elif market_open_session.startswith("NY"):
    open_hours = [16, 17]
else:
    open_hours = [3, 4]

# Days of history to calculate rolling average
rolling_window = st.sidebar.number_input("RVol Rolling Window (hours)", min_value=10, max_value=500, value=ROLLING_WINDOW)


# ------------------- DATA FETCHING -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol, start_date, end_date):
    t = Ticker(symbol, timeout=60)
    hist = t.history(period=f"{DAYS}d", interval="1h")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index()
        hist = hist.rename(columns={"symbol": "ticker"})
        hist = hist.dropna(subset=["volume", "date"])
        hist = hist[hist["volume"] > 0]
        hist["datetime"] = pd.to_datetime(hist["date"], errors="coerce", utc=True)
        hist = hist.dropna(subset=["datetime"])
        hist = hist.sort_values("datetime")
        # Filter by backtesting date range
        hist = hist[(hist["datetime"].dt.date >= start_date) & (hist["datetime"].dt.date <= end_date)]
        # Convert to GMT+3
        hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
        hist["datetime_gmt3"] = hist["datetime_gmt3"].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        # Calculate rolling average and rvol
        hist["avg_volume"] = hist["volume"].rolling(rolling_window).mean()
        hist["rvol"] = hist["volume"] / hist["avg_volume"]
        return hist
    else:
        return pd.DataFrame()

# ------------------- GAP-UP DETECTION -------------------
def detect_gap_up(df, open_hours):
    if df.empty:
        return None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    if df.empty or len(df["date_gmt3"].unique()) < 2:
        return None
    # Latest and previous day
    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - pd.Timedelta(days=1)
    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    if curr_open.empty or prev_open.empty:
        return None
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    if prev_mean == 0 or pd.isna(prev_mean):
        return None
    gap_ratio = curr_mean / prev_mean
    return {"latest_day": latest_day, "curr_rvol": curr_mean, "prev_rvol": prev_mean, "gap_ratio": gap_ratio}

# ------------------- DISTRIBUTION & STATS -------------------
import numpy as np

st.sidebar.header("Backtesting & Entry Session")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01").date())
entry_session = st.sidebar.selectbox(
    "Select Entry Session:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)

if entry_session.startswith("London"):
    open_hours = [10, 11]
elif entry_session.startswith("NY"):
    open_hours = [16, 17]
elif entry_session.startswith("Asian"):
    open_hours = [3, 4]
else:
    open_hours = [16, 17]

gap_ratios_all = []

st.title("RVol Gap-Up Backtesting Distribution")

for symbol in asset_symbols:
    df = fetch_rvol_data(symbol, start_date, end_date)
    gap_info = detect_gap_up(df, open_hours)
    if not gap_info:
        continue
    gap_ratios_all.append(gap_info["gap_ratio"])

if not gap_ratios_all:
    st.warning("No valid gap-up data found for selected assets and date range.")
else:
    gap_ratios_all = np.array(gap_ratios_all)
    st.subheader("Gap-Up Distribution")
    import plotly.express as px
    fig = px.histogram(gap_ratios_all, nbins=20, labels={"value": "Gap-Up Ratio"}, title="RVol Gap-Up Ratios Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Compute percentiles and stats
    percentile_50 = np.percentile(gap_ratios_all, 50)
    percentile_70 = np.percentile(gap_ratios_all, 70)
    percentile_90 = np.percentile(gap_ratios_all, 90)
    st.markdown(f"**50th percentile:** {percentile_50:.2f}")
    st.markdown(f"**70th percentile:** {percentile_70:.2f}")
    st.markdown(f"**90th percentile:** {percentile_90:.2f}")

    # Approximate win rate / RR logic
    st.subheader("Win Rate & Risk/Reward by Gap-Up Percentile")
    for perc, label in zip([50, 70, 90], ["Median", "High", "Extreme"]):
        wins = (gap_ratios_all >= np.percentile(gap_ratios_all, perc)).sum()
        total = len(gap_ratios_all)
        win_rate = wins / total * 100
        rr = np.mean(gap_ratios_all[gap_ratios_all >= np.percentile(gap_ratios_all, perc)]) / np.mean(gap_ratios_all)
        st.markdown(f"**{label} Gap-Ups ({perc}th percentile):** Win Rate = {win_rate:.1f}%, Avg RR = {rr:.2f}")