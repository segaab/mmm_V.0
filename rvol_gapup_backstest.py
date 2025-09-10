# ------------------- CHUNK 1 -------------------
import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import timedelta, datetime
import numpy as np
import plotly.graph_objs as go

# ------------------- Ticker Map -------------------
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
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6N=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "^DJI",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "COPPER - COMMODITY EXCHANGE INC.":"HG=F"
}

# Remove unsupported symbols (optional)
TICKER_MAP = {k: v for k, v in TICKER_MAP.items() if v not in ["^N225", "DX-Y.NYB"]}

# List of all asset symbols
asset_symbols = list(TICKER_MAP.values())
TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}

# ------------------- Streamlit Setup -------------------
st.title("RVol Gap-Up Distribution Dashboard")

DAYS = 730  # 2 years for history
ROLLING_WINDOW = 120

# Market open options for gap detection
st.sidebar.header("Market Open Settings")
market_open = st.sidebar.selectbox(
    "Select Market Open Window:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)
# Determine hours
if market_open.startswith("London"):
    open_hours = [10, 11]
elif market_open.startswith("NY"):
    open_hours = [16, 17]
else:
    open_hours = [3, 4]

# ------------------- RVol Gap-Up Filter -------------------
def detect_gap_up(df, open_hours):
    """Detects gap-ups and returns gap ratio for latest day vs previous."""
    if df.empty:
        return None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    if df.empty or df.shape[0] < 2:
        return None
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
    return gap_ratio



# ------------------- CHUNK 2 -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol):
    """Fetch historical 1h data for an asset and calculate RVol."""
    try:
        t = Ticker(symbol, timeout=60)
        hist = t.history(period=f"{DAYS}d", interval="1h")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.reset_index()
            hist = hist.rename(columns={"symbol": "ticker", "date": "datetime"})
            hist = hist.dropna(subset=["volume", "datetime"])
            hist = hist[hist["volume"] > 0]
            hist["datetime"] = pd.to_datetime(hist["datetime"], errors="coerce")
            hist = hist.dropna(subset=["datetime"])
            hist = hist.sort_values("datetime")
            # Convert to GMT+3
            hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
            hist["datetime_gmt3"] = hist["datetime_gmt3"].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
            # Calculate rolling average and RVol
            hist["avg_volume"] = hist["volume"].rolling(ROLLING_WINDOW).mean()
            hist["rvol"] = hist["volume"] / hist["avg_volume"]
            return hist
        else:
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()

# ------------------- Compute Gap-Up Distribution -------------------
gap_ratios_all = {}

for symbol in asset_symbols:
    asset_name = TICKER_TO_NAME.get(symbol, symbol)
    df = fetch_rvol_data(symbol)
    gap_ratio = detect_gap_up(df, open_hours)
    if gap_ratio is not None:
        gap_ratios_all[asset_name] = gap_ratio

if not gap_ratios_all:
    st.warning("No gap-up data available for the selected assets.")
else:
    st.subheader("Gap-Up Distribution (Latest Day)")
    gap_df = pd.DataFrame(list(gap_ratios_all.items()), columns=["Asset", "Gap Ratio"])
    gap_df = gap_df.sort_values("Gap Ratio", ascending=False)
    st.dataframe(gap_df)

    # Histogram of gap ratios
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=gap_df["Gap Ratio"], nbinsx=20, marker_color="blue"))
    fig_hist.update_layout(
        title="Gap-Up Ratio Distribution Across Assets",
        xaxis_title="Gap Ratio",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)



# ------------------- CHUNK 3 -------------------
st.subheader("Gap-Up Percentile Analysis & Trade Statistics")

# Calculate percentiles for each asset
percentiles = {}
for asset, ratio in gap_ratios_all.items():
    all_ratios = fetch_rvol_data(TICKER_MAP[asset])["rvol"].dropna()
    if all_ratios.empty:
        continue
    percentile = (all_ratios < ratio).mean() * 100  # percentile of latest gap
    percentiles[asset] = percentile

percentile_df = pd.DataFrame(list(percentiles.items()), columns=["Asset", "Gap Percentile"])
percentile_df = percentile_df.sort_values("Gap Percentile", ascending=False)
st.dataframe(percentile_df)

# Compute simple win-rate and risk-reward estimate (mock logic)
win_rates = {}
rr_ratios = {}
for asset, pct in percentiles.items():
    # Example: Higher percentile -> higher win probability, but lower RR
    win_rate = min(max(0.3 + pct / 150, 0), 1)  # mock formula
    rr_ratio = max(1.5 - pct / 100, 0.2)       # mock formula
    win_rates[asset] = win_rate
    rr_ratios[asset] = rr_ratio

stats_df = pd.DataFrame({
    "Asset": list(win_rates.keys()),
    "Win Rate": [f"{v*100:.1f}%" for v in win_rates.values()],
    "Risk/Reward": [f"{v:.2f}" for v in rr_ratios.values()]
}).sort_values("Win Rate", ascending=False)

st.subheader("Gap-Up Trade Stats by Asset")
st.dataframe(stats_df)

# Optional: visualize win rate vs RR
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=[rr_ratios[a] for a in stats_df["Asset"]],
    y=[win_rates[a]*100 for a in stats_df["Asset"]],
    mode="markers+text",
    text=stats_df["Asset"],
    textposition="top center",
    marker=dict(size=10, color="orange")
))
fig_scatter.update_layout(
    title="Win Rate vs Risk/Reward by Asset",
    xaxis_title="Risk/Reward Ratio",
    yaxis_title="Win Rate (%)",
    height=400
)
st.plotly_chart(fig_scatter, use_container_width=True)