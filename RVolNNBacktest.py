import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import concurrent.futures
from functools import partial

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
# Data Fetching with Improved Error Handling
# -------------------
@st.cache_data(show_spinner=False)
def get_data(ticker_symbol, start_date, end_date, retries=3):
    """Fetch historical data with improved error handling and retries"""
    for attempt in range(retries):
        try:
            ticker = Ticker(ticker_symbol, timeout=30)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1h"
            )
            
            if isinstance(df, dict) and 'error' in df:
                st.warning(f"Error fetching {ticker_symbol}: {df['error']}")
                return None
                
            if df.empty:
                st.warning(f"No data returned for {ticker_symbol}")
                return None
                
            # Reset index if it's a multi-index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                
            # Ensure required columns exist
            if 'date' not in df.columns or 'volume' not in df.columns:
                st.warning(f"Missing required columns in data for {ticker_symbol}")
                return None
                
            # Process the dataframe
            df.rename(columns={"symbol": "ticker"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["date"], utc=True)
            
            # Convert to GMT+3 for consistent timezone handling
            df["datetime_gmt3"] = df["datetime"] + pd.Timedelta(hours=3)
            df.set_index("datetime_gmt3", inplace=True)
            
            # Select only needed columns
            df = df[["open", "high", "low", "close", "volume"]]
            
            # Remove rows with zero volume
            df = df[df["volume"] > 0]
            
            if df.empty:
                st.warning(f"No valid data with positive volume for {ticker_symbol}")
                return None
                
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"Retry {attempt+1}/{retries} for {ticker_symbol}: {str(e)}")
                time.sleep(1)  # Add delay between retries
            else:
                st.error(f"Failed to fetch data for {ticker_symbol}: {str(e)}")
                return None

# -------------------
# RVol Calculation
# -------------------
def calculate_rvol(df, window=5):
    """Calculate relative volume with proper handling of edge cases"""
    if df is None or df.empty:
        return None
        
    df = df.copy()
    
    # Ensure data is sorted chronologically
    df = df.sort_index()
    
    # Calculate average volume using rolling window
    df["avg_vol"] = df["volume"].rolling(window=window).mean()
    
    # Calculate relative volume (RVol)
    df["rvol"] = df["volume"] / df["avg_vol"]
    
    # Handle NaN values (first 'window' rows will have NaN avg_vol)
    df = df.dropna(subset=["rvol"])
    
    return df

# -------------------
# Gap-Up Detection (Improved)
# -------------------
def detect_gap_up(df, session_hours, threshold=2.0):
    """Detect gap-up in relative volume between consecutive days for specified session hours"""
    if df is None or df.empty:
        return None, None, None
        
    df = df.copy()
    
    # Extract date and hour information
    df["date"] = df.index.date
    df["hour"] = df.index.hour
    
    # Get unique dates in descending order
    dates = sorted(df["date"].unique(), reverse=False)
    
    # Need at least 2 dates for comparison
    if len(dates) < 2:
        return None, None, None
    
    # Store results for each pair of consecutive days
    results = []
    
    # Iterate through consecutive pairs of dates
    for i in range(1, len(dates)):
        current_day = dates[i]
        prev_day = dates[i-1]
        
        # Get RVol data for session hours on both days
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
        
    # Return the most recent gap-up that exceeds the threshold
    for date, ratio, curr, prev in reversed(results):
        if ratio >= threshold:
            return ratio, curr, prev
            
    # If no gap-up found, return the most recent result
    date, ratio, curr, prev = results[-1]
    return ratio, curr, prev

# -------------------
# Backtesting Logic
# -------------------
def backtest_asset(symbol, session_hours, start_date, end_date, threshold=2.0, window=5, holding_period=5):
    """Backtest a single asset for RVol gap-up strategy"""
    # Fetch historical data
    df = get_data(symbol, start_date, end_date)
    if df is None or df.empty:
        return None, symbol

    # Calculate relative volume
    df = calculate_rvol(df, window)
    if df is None or df.empty:
        return None, symbol

    # Find gap-ups
    trades = []
    
    # Extract all dates in chronological order
    dates = sorted(df.index.date.unique())
    
    for i in range(1, len(dates)-holding_period):
        # For each date, check if there was a gap-up
        temp_df = df[df.index.date <= dates[i]].copy()
        gap_ratio, curr_mean, prev_mean = detect_gap_up(temp_df, session_hours, threshold)
        
        if gap_ratio is not None and gap_ratio >= threshold:
            # If gap-up detected, simulate entry at close of the gap-up day
            entry_date = dates[i]
            entry_price = df[df.index.date == entry_date]["close"].iloc[-1]
            
            # Exit after holding_period days
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

# -------------------
# Multi-Asset Backtesting
# -------------------
selected_assets = st.sidebar.multiselect(
    "Select Assets", options=list(TICKER_MAP.keys()), default=["Gold", "EUR/USD", "Crude Oil"]
)

if st.sidebar.button("Run Backtest"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    total_assets = len(selected_assets)
    
    # Process one asset at a time instead of using ThreadPoolExecutor
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
                t["asset"] = [k for k, v in TICKER_MAP.items() if v == symbol][0]
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
            # Calculate sharpe ratio (simplified)
            if len(df_results) > 1:
                sharpe = df_results["return_pct"].mean() / df_results["return_pct"].std()
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # Trades table
        st.subheader("Trade Details")
        st.dataframe(df_results.sort_values("entry_date", ascending=False))
        
        # Distribution of returns
        st.subheader("Return Distribution")
        fig = px.histogram(
            df_results, x="return_pct", nbins=20, 
            title="Return Distribution (%)",
            color_discrete_sequence=["#3366CC"]
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns by asset
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
