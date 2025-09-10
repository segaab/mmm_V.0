import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import timedelta, datetime
import numpy as np
import plotly.graph_objs as go
from concurrent.futures import ThreadPoolExecutor

# ----------------- Asset Mapping -----------------
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

# Reverse mapping for display
TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}

# ----------------- Streamlit Page -----------------
st.title("RVol Gap-Up Backtester")

# Sidebar inputs for backtesting
st.sidebar.header("Backtesting Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01").date())
entry_session = st.sidebar.selectbox(
    "Select Entry Session",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)
rolling_window_hours = st.sidebar.number_input(
    "RVol Rolling Window (hours)", min_value=10, max_value=500, value=120, step=10
)

# Map session to hours
if entry_session.startswith("London"):
    open_hours = [10, 11]
elif entry_session.startswith("NY"):
    open_hours = [16, 17]
elif entry_session.startswith("Asian"):
    open_hours = [3, 4]
else:
    open_hours = [16, 17]

# ----------------- Utilities -----------------
def detect_gap_up(df, open_hours):
    if df.empty:
        return False, None, None, None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    if df.empty:
        return False, None, None, None
    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - pd.Timedelta(days=1)
    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    if curr_open.empty or prev_open.empty:
        return False, None, None, None
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    gap_ratio = curr_mean / prev_mean if prev_mean > 0 else 0
    return gap_ratio > 1.0, curr_mean, prev_mean, gap_ratio  # Returns True if any gap up occurs

# ----------------- RVol Data Fetching -----------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol):
    """
    Fetches historical hourly data and computes RVol for a given symbol.
    Returns a DataFrame with datetime_gmt3, volume, and rvol.
    """
    try:
        t = Ticker(symbol, timeout=60)
        hist = t.history(period="730d", interval="1h")
        if hist.empty:
            return pd.DataFrame()
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index()
        hist = hist.rename(columns={"symbol": "ticker"})
        hist = hist.dropna(subset=["volume", "date"])
        hist = hist[hist["volume"] > 0]
        hist["datetime"] = pd.to_datetime(hist["date"], errors="coerce", utc=True)
        hist = hist.dropna(subset=["datetime"])
        # Filter by backtesting date range
        hist = hist[(hist["datetime"] >= pd.to_datetime(start_date)) & (hist["datetime"] <= pd.to_datetime(end_date))]
        hist = hist.sort_values("datetime")
        # Convert to GMT+3
        hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
        hist["datetime_gmt3"] = hist["datetime_gmt3"].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        # Calculate rolling average and rvol
        hist["avg_volume"] = hist["volume"].rolling(rolling_window_hours).mean()
        hist["rvol"] = hist["volume"] / hist["avg_volume"]
        return hist
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


# ----------------- Multi-Asset Threading -----------------
def fetch_all_assets(symbols):
    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_rvol_data, sym): sym for sym in symbols}
        for future in futures:
            sym = futures[future]
            try:
                results[sym] = future.result()
            except Exception as e:
                st.warning(f"Failed to fetch {sym}: {e}")
                results[sym] = pd.DataFrame()
    return results


# ----------------- Asset List -----------------
asset_symbols = list(TICKER_MAP.values())
asset_data = fetch_all_assets(asset_symbols)
st.success(f"Fetched RVol data for {len(asset_data)} assets.")

# ----------------- Gap-Up Detection -----------------
def detect_gap_up(df, entry_hours):
    """
    Detects gap-up based on RVol during the selected entry session hours.
    Returns gap_ratio, current_mean, previous_mean.
    """
    if df.empty:
        return None, None, None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour

    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - pd.Timedelta(days=1)

    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(entry_hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(entry_hours))]["rvol"]

    if curr_open.empty or prev_open.empty:
        return None, curr_open.mean() if not curr_open.empty else None, prev_open.mean() if not prev_open.empty else None

    prev_mean = prev_open.mean()
    if prev_mean == 0 or pd.isna(prev_mean):
        return None, curr_open.mean(), prev_mean

    gap_ratio = curr_open.mean() / prev_mean
    return gap_ratio, curr_open.mean(), prev_mean


# ----------------- Backtesting and Distribution -----------------
gap_ratios = []
for sym, df in asset_data.items():
    gap_ratio, curr_mean, prev_mean = detect_gap_up(df, entry_hours)
    if gap_ratio is not None:
        gap_ratios.append({"symbol": sym, "gap_ratio": gap_ratio, "curr_rvol": curr_mean, "prev_rvol": prev_mean})

gap_df = pd.DataFrame(gap_ratios)
if gap_df.empty:
    st.warning("No gap-ups detected in the selected date range and session.")
else:
    st.subheader("Gap-Up Distribution")
    st.write(f"Total gap-ups detected: {len(gap_df)}")
    st.write(gap_df.describe())

    # Plot histogram
    import plotly.express as px
    fig = px.histogram(gap_df, x="gap_ratio", nbins=20, title="RVol Gap-Up Ratio Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Compute percentile stats
    for percentile in [0.5, 0.7, 0.8, 0.9]:
        val = gap_df["gap_ratio"].quantile(percentile)
        st.write(f"{int(percentile*100)}th percentile gap ratio: {val:.2f}")