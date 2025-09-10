# ------------------- CHUNK 1 -------------------
import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import datetime, timedelta
import plotly.graph_objs as go

# ------------------- Asset Mapping -------------------
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
asset_symbols = list(TICKER_MAP.values())

# ------------------- Streamlit Layout -------------------
st.title("RVol Gap-Up Backtester")

# ------------------- Backtesting Settings -------------------
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime(2023, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.today())

entry_session = st.sidebar.selectbox(
    "Select Entry Session:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)

rvol_window = st.sidebar.number_input("RVol Rolling Window (hours)", min_value=10, max_value=500, value=120, step=10)

# Map session to hours
if entry_session.startswith("London"):
    entry_hours = [10, 11]
elif entry_session.startswith("NY"):
    entry_hours = [16, 17]
elif entry_session.startswith("Asian"):
    entry_hours = [3, 4]
else:
    entry_hours = [16, 17]





# ------------------- CHUNK 2 -------------------

# ------------------- Functions -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol, start_date, end_date):
    """
    Fetch hourly data for a symbol from YahooQuery and calculate RVol.
    """
    t = Ticker(symbol, timeout=60)
    hist = t.history(period="max", interval="1h")
    if hist.empty:
        return pd.DataFrame()
    
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index()
    hist = hist.rename(columns={"symbol": "ticker", "date": "datetime"})
    hist = hist.dropna(subset=["volume", "datetime"])
    hist = hist[hist["volume"] > 0]
    hist["datetime"] = pd.to_datetime(hist["datetime"], errors="coerce", utc=True)
    hist = hist.dropna(subset=["datetime"])
    hist = hist.sort_values("datetime")
    
    # Filter by backtesting date range
    hist = hist[(hist["datetime"] >= pd.to_datetime(start_date)) & (hist["datetime"] <= pd.to_datetime(end_date))]
    
    # Convert to GMT+3
    hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
    hist["datetime_gmt3_dt"] = hist["datetime_gmt3"]
    
    # RVol calculation
    hist["avg_volume"] = hist["volume"].rolling(rvol_window).mean()
    hist["rvol"] = hist["volume"] / hist["avg_volume"]
    
    return hist

def detect_gap_up(df, hours):
    """
    Detect RVol gap-ups in specified session hours.
    Returns: gap_ratio, current_mean_rvol, previous_mean_rvol
    """
    if df.empty:
        return None, None, None
    df = df.copy()
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    
    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - timedelta(days=1)
    
    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(hours))]["rvol"]
    
    if curr_open.empty or prev_open.empty:
        return None, None, None
    
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    
    if prev_mean == 0 or pd.isna(prev_mean):
        return None, curr_mean, prev_mean
    
    gap_ratio = curr_mean / prev_mean
    return gap_ratio, curr_mean, prev_mean






# ------------------- CHUNK 3 -------------------

# ------------------- Main Backtesting -------------------
st.header("RVol Gap-Up Backtesting Results")

gap_results = []

for symbol in asset_symbols:
    asset_name = TICKER_TO_NAME.get(symbol, symbol)
    df = fetch_rvol_data(symbol, start_date, end_date)
    if df.empty:
        st.warning(f"No data for {asset_name} ({symbol}) in selected date range.")
        continue

    gap_ratio, curr_mean, prev_mean = detect_gap_up(df, entry_hours)
    if gap_ratio is None:
        continue

    gap_results.append({
        "symbol": symbol,
        "asset_name": asset_name,
        "gap_ratio": gap_ratio,
        "curr_rvol": curr_mean,
        "prev_rvol": prev_mean
    })

# Convert results to DataFrame
gap_df = pd.DataFrame(gap_results)
if gap_df.empty:
    st.info("No gap-ups detected for selected assets and date range.")
else:
    st.subheader("Gap-Up Distribution Stats")
    
    # Display statistics
    st.write(gap_df.describe())

    # Plot distribution
    import plotly.express as px
    fig = px.histogram(gap_df, x="gap_ratio", nbins=20, title="RVol Gap-Up Ratio Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate approximate win rate and RR (for illustrative purposes, assuming hypothetical return)
    # Here we simulate win if gap_ratio > median
    median_ratio = gap_df["gap_ratio"].median()
    gap_df["win"] = gap_df["gap_ratio"] > median_ratio
    win_rate = gap_df["win"].mean() * 100
    rr_ratio = gap_df["gap_ratio"].mean() / gap_df["gap_ratio"].median() if median_ratio != 0 else None
    
    st.metric("Approx. Win Rate (%)", f"{win_rate:.1f}")
    st.metric("R/R (mean/median gap ratio)", f"{rr_ratio:.2f}" if rr_ratio else "N/A")
    
    st.dataframe(gap_df.sort_values("gap_ratio", ascending=False).reset_index(drop=True))