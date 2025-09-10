import streamlit as st
import pandas as pd
from yahooquery import Ticker
from datetime import timedelta
import json
from queue import Queue
from threading import Thread
import plotly.graph_objs as go

# ------------------- Ticker maps (assets only, ETFs removed) -------------------
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

# Remove ^N225 and DX-Y.NYB
TICKER_MAP = {k: v for k, v in TICKER_MAP.items() if v not in ["^N225", "DX-Y.NYB"]}

# All unique asset symbols
asset_symbols = list(TICKER_MAP.values())

# Load sector mapping from asset_category_map.json
with open("asset_category_map.json", "r") as f:
    ASSET_CATEGORY_MAP = json.load(f)
ASSET_TO_SECTOR = {asset: sector for sector, assets in ASSET_CATEGORY_MAP.items() for asset in assets}
TICKER_TO_NAME = {v: k for k, v in TICKER_MAP.items()}

# Constants
DAYS = 730
ROLLING_WINDOW = 120

# ------------------- Streamlit UI -------------------
st.title("RVol Gap-Up Backtesting Dashboard")
if st.button("Rerun"):
    st.rerun()

market_open = st.sidebar.selectbox(
    "Select Market Open Window:",
    ["London (10:00-11:00)", "NY (16:00-17:00)", "Asian (3:00-4:00)"]
)

if market_open.startswith("London"):
    open_hours = [10, 11]
elif market_open.startswith("NY"):
    open_hours = [16, 17]
elif market_open.startswith("Asian"):
    open_hours = [3, 4]
else:
    open_hours = [16, 17]

# ------------------- RVol Fetch -------------------
@st.cache_data(show_spinner=True)
def fetch_rvol_data(symbol):
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
        # Convert to GMT+3
        hist["datetime_gmt3"] = hist["datetime"] + timedelta(hours=3)
        hist["datetime_gmt3"] = hist["datetime_gmt3"].dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
        # RVol calculation
        hist["avg_volume"] = hist["volume"].rolling(ROLLING_WINDOW).mean()
        hist["rvol"] = hist["volume"] / hist["avg_volume"]
        return hist
    else:
        return pd.DataFrame()


# ------------------- Gap-Up Detection -------------------
def detect_gap_up(df, open_hours):
    if df.empty:
        return None, None, None
    df = df.copy()
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    if df.empty:
        return None, None, None
    latest_day = df.iloc[0]["date_gmt3"]
    prev_day = latest_day - pd.Timedelta(days=1)
    # RVol for open hours
    curr_open = df[(df["date_gmt3"] == latest_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    prev_open = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"]
    if curr_open.empty or prev_open.empty:
        return None, None, None
    curr_mean = curr_open.mean()
    prev_mean = prev_open.mean()
    if prev_mean == 0 or pd.isna(prev_mean):
        return None, curr_mean, prev_mean
    gap_ratio = curr_mean / prev_mean
    return gap_ratio, curr_mean, prev_mean

# ------------------- Multi-Threaded Backtesting -------------------
def process_asset(symbol, result_queue):
    df = fetch_rvol_data(symbol)
    if df.empty:
        result_queue.put((symbol, None, None, None))
        return
    gap_ratio, curr_rvol, prev_rvol = detect_gap_up(df, open_hours)
    if gap_ratio is not None:
        # Calculate historical percentile of current gap_ratio
        historical_ratios = []
        df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
        df = df.dropna(subset=["datetime_gmt3_dt"])
        df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
        unique_dates = sorted(df["date_gmt3"].unique())
        for i in range(1, len(unique_dates)):
            day = unique_dates[i]
            prev_day = unique_dates[i-1]
            day_r = df[(df["date_gmt3"] == day) & (df["hour_gmt3"].isin(open_hours))]["rvol"].mean()
            prev_day_r = df[(df["date_gmt3"] == prev_day) & (df["hour_gmt3"].isin(open_hours))]["rvol"].mean()
            if pd.notna(day_r) and pd.notna(prev_day_r) and prev_day_r != 0:
                historical_ratios.append(day_r / prev_day_r)
        if historical_ratios:
            percentile = (sum(r <= gap_ratio for r in historical_ratios) / len(historical_ratios)) * 100
        else:
            percentile = None
    else:
        percentile = None
    result_queue.put((symbol, gap_ratio, curr_rvol, prev_rvol, percentile))

# ------------------- Run Multi-Threaded Backtest -------------------
def run_backtest(assets):
    threads = []
    result_queue = Queue()
    for symbol in assets:
        thread = Thread(target=process_asset, args=(symbol, result_queue))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results

# ------------------- Run Backtest and Display -------------------
st.header("RVol Gap-Up Backtesting Results")

# Run backtest on all selected assets
backtest_results = run_backtest(asset_symbols)

# Prepare DataFrame for display
results_df = pd.DataFrame(backtest_results, columns=["Symbol", "Gap_Ratio", "Curr_RVol", "Prev_RVol", "Percentile"])
results_df = results_df.dropna(subset=["Gap_Ratio"])
results_df = results_df.sort_values("Percentile", ascending=False)

st.subheader("Gap-Up Percentiles")
st.dataframe(results_df[["Symbol", "Gap_Ratio", "Curr_RVol", "Prev_RVol", "Percentile"]])

# ------------------- Distribution Plot -------------------
import plotly.express as px
fig_dist = px.histogram(results_df, x="Percentile", nbins=20, title="Distribution of Gap-Up Percentiles")
st.plotly_chart(fig_dist, use_container_width=True)

# ------------------- Win Rate & RR -------------------
# Example logic: consider percentile > 80 as "high gap"
high_gap = results_df[results_df["Percentile"] >= 80]
win_rate = len(high_gap) / len(results_df) if len(results_df) > 0 else 0
# For illustration, reward-to-risk ratio set as gap_ratio / 1.0
rr = high_gap["Gap_Ratio"].mean() if not high_gap.empty else 0
st.markdown(f"**High Gap (>80th percentile) Win Rate:** {win_rate:.2%}")
st.markdown(f"**Average Reward-to-Risk (R/R) for High Gap:** {rr:.2f}")

# ------------------- Plot Latest Day RVol per Asset -------------------
st.subheader("Latest Day Hourly RVol")
for symbol in asset_symbols:
    df = fetch_rvol_data(symbol)
    if df.empty:
        continue
    df["datetime_gmt3_dt"] = pd.to_datetime(df["datetime_gmt3"], errors="coerce")
    df = df.dropna(subset=["datetime_gmt3_dt"])
    df = df.sort_values("datetime_gmt3_dt", ascending=False)
    df["hour_gmt3"] = df["datetime_gmt3_dt"].dt.hour
    df["date_gmt3"] = df["datetime_gmt3_dt"].dt.date
    latest_day = df.iloc[0]["date_gmt3"]
    day_df = df[df["date_gmt3"] == latest_day].copy()
    if day_df.empty:
        continue
    chart_df = day_df.set_index("hour_gmt3")[["rvol"]].sort_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["rvol"], name=f"{symbol} RVol", marker_color="blue"))
    percentile_70 = df["rvol"].quantile(0.7)
    fig.add_hline(y=percentile_70, line_width=3, line_dash="dash", line_color="red",
                  annotation_text="70th percentile", annotation_position="top right")
    fig.update_layout(
        title=f"{symbol} — {latest_day}",
        xaxis_title="Hour of Day (GMT+3)",
        yaxis_title="RVol",
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[str(h) for h in range(24)]),
        yaxis=dict(rangemode="tozero"),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True, key=f"rvol-{symbol}")