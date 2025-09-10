import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import datetime
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(layout="wide", page_title="RVol Gap-Up Backtester")

st.title("RVol Gap-Up Strategy Backtester")
st.markdown("""
This dashboard backtests the Relative Volume (RVol) Gap-Up trading strategy using hourly data.
The strategy identifies volume spikes at session opens and trades based on the subsequent candle pattern.
""")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("Strategy Parameters")

# Batch asset mapping
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
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": "^N225",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "COPPER - COMMODITY EXCHANGE INC.":"HG=F"
}

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = col2.date_input("End Date", datetime.date.today())

# RVol parameters
rvol_window = st.sidebar.slider("RVol Window (hours)", 1, 24, 5)
gap_threshold = st.sidebar.slider("Gap-Up Threshold", 1.0, 5.0, 2.0, 0.1)

# Risk management
atr_window = st.sidebar.slider("ATR Window (hours)", 5, 48, 14)
sl_multiplier = st.sidebar.slider("Stop Loss (x ATR)", 0.5, 3.0, 1.5, 0.1)
tp_multiplier = st.sidebar.slider("Take Profit (x ATR)", 0.5, 5.0, 2.0, 0.1)

# Sessions
sessions = st.sidebar.multiselect(
    "Sessions to Test", ["Asian", "London", "New York"], ["Asian", "London", "New York"]
)

run_backtest = st.sidebar.button("Run Backtest")

# ------------------- Data Fetching Function -------------------
def get_data(ticker_symbol, start_date, end_date):
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        ticker = Ticker(ticker_symbol)
        df = ticker.history(interval="1h", start=start_str, end=end_str)

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index().set_index('date')
        
        df.columns = [c.lower() for c in df.columns]
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} missing from {ticker_symbol} data")
        
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        return None, str(e)


# ------------------- RVol & ATR Calculations -------------------
def calculate_rvol(df, window=rvol_window):
    df = df.copy()
    df['avg_volume'] = df['volume'].rolling(window=window, min_periods=1).mean()
    df['rvol'] = df['volume'] / df['avg_volume']
    return df

def calculate_atr(df, window=atr_window):
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=window, min_periods=1).mean()
    return df

# ------------------- Gap-Up Detection -------------------
def detect_gap_up(df, gap_threshold=gap_threshold):
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['gap_up'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df['gap_signal'] = df['gap_up'] >= gap_threshold
    return df

# ------------------- Signal Generation -------------------
def generate_signals(df):
    df = df.copy()
    df['rvol_signal'] = df['rvol'] >= 1.0
    df['long_signal'] = df['rvol_signal'] & df['gap_signal']
    df['short_signal'] = False  # Optional for gap-down
    return df

# ------------------- Backtesting Function -------------------
def backtest_asset(symbol, start_date, end_date, rvol_window, gap_threshold, atr_window, sl_multiplier, tp_multiplier, queue):
    df = get_data(symbol, start_date, end_date)
    if df is None or isinstance(df, tuple):
        queue.put((symbol, None, f"Data fetch failed: {df[1]}"))
        return
    
    df = calculate_rvol(df, rvol_window)
    df = calculate_atr(df, atr_window)
    df = detect_gap_up(df, gap_threshold)
    df = generate_signals(df)

    trades = []
    for idx, row in df.iterrows():
        if row['long_signal']:
            entry = row['open']
            sl = entry - row['atr'] * sl_multiplier
            tp = entry + row['atr'] * tp_multiplier
            trades.append({'date': idx, 'entry': entry, 'sl': sl, 'tp': tp, 'type': 'long'})
    queue.put((symbol, df, trades))

import threading
import queue

# ------------------- Multi-Asset Backtesting -------------------
def run_backtest_multi_assets(assets, start_date, end_date, rvol_window, gap_threshold, atr_window, sl_multiplier, tp_multiplier):
    q = queue.Queue()
    threads = []

    # Start threads for each asset
    for symbol in assets:
        t = threading.Thread(
            target=backtest_asset,
            args=(symbol, start_date, end_date, rvol_window, gap_threshold, atr_window, sl_multiplier, tp_multiplier, q)
        )
        t.start()
        threads.append(t)

    # Wait for threads to finish
    for t in threads:
        t.join()

    # Collect results
    results = {}
    while not q.empty():
        symbol, df, trades = q.get()
        results[symbol] = {'data': df, 'trades': trades}

    return results

# ------------------- Streamlit Dashboard -------------------
if run_backtest:
    with st.spinner("Running multi-asset backtest..."):
        assets = list(TICKER_MAP.values())  # Run all assets
        results = run_backtest_multi_assets(
            assets,
            start_date,
            end_date,
            rvol_window,
            gap_threshold,
            atr_window,
            sl_multiplier,
            tp_multiplier
        )

    for symbol, result in results.items():
        st.subheader(f"Asset: {symbol}")
        df = result['data']
        trades = result['trades']

        if df is None or trades is None:
            st.error(f"Data or trades not available for {symbol}")
            continue

        st.write(f"Number of trades: {len(trades)}")

        # Display trades
        trade_df = pd.DataFrame(trades)
        if not trade_df.empty:
            trade_df['pnl'] = (trade_df['tp'] - trade_df['entry']) / trade_df['entry']
            st.dataframe(trade_df, use_container_width=True)

        # Plot price + RVol
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Price", "Relative Volume"])
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rvol'], mode='lines', name='RVol'), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)