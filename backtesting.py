# ===== Chunk 1/3 =====
import os
import logging
import time
import threading
from typing import Dict, Tuple, List, Any
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
from yahooquery import Ticker

import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Asset object ---
class Asset:
    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker

assets_list = [
    Asset("GOLD - COMMODITY EXCHANGE INC.", "GC=F"),
    Asset("SILVER - COMMODITY EXCHANGE INC.", "SI=F"),
    Asset("EURO FX - CHICAGO MERCANTILE EXCHANGE", "6E=F"),
    Asset("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "6J=F"),
    Asset("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "6B=F"),
    Asset("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "6C=F"),
    Asset("AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "6A=F"),
    Asset("SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "6S=F"),
    Asset("S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "ES=F"),
    Asset("NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "NQ=F"),
    Asset("DOW JONES INDUSTRIAL AVERAGE - CHICAGO MERCANTILE EXCHANGE", "YM=F"),
    Asset("CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE", "CL=F"),
    Asset("NATURAL GAS - NEW YORK MERCANTILE EXCHANGE", "NG=F"),
    Asset("COPPER - COMMODITY EXCHANGE INC.", "HG=F"),
    Asset("PLATINUM - NEW YORK MERCANTILE EXCHANGE", "PL=F"),
    Asset("PALLADIUM - NEW YORK MERCANTILE EXCHANGE", "PA=F"),
]

# ------------------------------
# Data fetching helpers
# ------------------------------
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    """Fetch COT rows for a given market name from the public Socrata endpoint."""
    logger.info("Fetching COT data for %s", market_name)
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
                df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", pd.Series()), errors="coerce")
                # derive nets safely
                try:
                    df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
                    df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
                except Exception:
                    df["commercial_net"] = 0.0
                    df["non_commercial_net"] = 0.0
                return df.sort_values("report_date").reset_index(drop=True)
            else:
                logger.warning("No COT data for %s", market_name)
                return pd.DataFrame()
        except Exception as e:
            logger.error("Error fetching COT data for %s: %s", market_name, e)
            attempt += 1
            time.sleep(1 + attempt)
    logger.error("Failed fetching COT data for %s after %d attempts.", market_name, max_attempts)
    return pd.DataFrame()

def fetch_price_data_yahoo(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    """Fetch daily price history from yahooquery for a given ticker and date range."""
    logger.info("Fetching Yahoo data for %s from %s to %s", ticker, start_date, end_date)
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
                logger.warning("No price data for %s", ticker)
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                try:
                    hist = hist.loc[ticker]
                except Exception:
                    hist = hist.reset_index(level=0, drop=True)
            hist = hist.reset_index()
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            else:
                hist.index = pd.to_datetime(hist.index)
                hist = hist.reset_index().rename(columns={"index": "date"})
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching Yahoo data for %s: %s", ticker, e)
            attempt += 1
            time.sleep(1 + attempt)
    logger.error("Failed fetching Yahoo data for %s after %d attempts.", ticker, max_attempts)
    return pd.DataFrame()

# ===== Chunk 2/3 =====
# ------------------------------
# Threaded fetching
# ------------------------------
def fetch_all_data(selected_asset: Asset, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cot_df, price_df = None, None

    def fetch_cot():
        nonlocal cot_df
        cot_df = fetch_cot_data(selected_asset.name)

    def fetch_price():
        nonlocal price_df
        price_df = fetch_price_data_yahoo(selected_asset.ticker, start_date, end_date)

    threads = []
    for func in [fetch_cot, fetch_price]:
        thread = threading.Thread(target=func)
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()

    return cot_df, price_df

# ------------------------------
# Health gauge calculation
# ------------------------------
def calculate_health_gauge(cot_df: pd.DataFrame) -> float:
    """Example: health gauge from commercial net position."""
    if cot_df.empty:
        return 0.0
    latest = cot_df.iloc[-1]
    net_ratio = latest["commercial_net"] / max(latest["open_interest_all"], 1)
    return float(np.clip(net_ratio, -1, 1))

# ------------------------------
# Price bands / ATR / RVol
# ------------------------------
def calculate_price_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    close_col = "close"
    if close_col not in df.columns:
        raise KeyError(f"{close_col} not in price data")
    df["ma"] = df[close_col].rolling(window, min_periods=1).mean()
    df["std"] = df[close_col].rolling(window, min_periods=1).std()
    df["upper_band"] = df["ma"] + 2 * df["std"]
    df["lower_band"] = df["ma"] - 2 * df["std"]
    df["returns"] = df[close_col].pct_change().fillna(0.0)
    df["atr"] = df["close"].diff().abs().rolling(window).mean()
    df["rvol"] = df["returns"].rolling(window).std()
    return df

# ------------------------------
# Signal generation
# ------------------------------
def generate_signals(df: pd.DataFrame, buy_threshold: float = 0.0, sell_threshold: float = 0.0,
                     atr_lookback: int = 14, rvol_lookback: int = 14) -> pd.DataFrame:
    df = calculate_price_bands(df)
    df["position"] = 0
    df.loc[df["close"] < df["lower_band"] * (1 - buy_threshold), "position"] = 1
    df.loc[df["close"] > df["upper_band"] * (1 + sell_threshold), "position"] = -1
    df["atr_signal"] = df["atr"].rolling(atr_lookback).mean()
    df["rvol_signal"] = df["rvol"].rolling(rvol_lookback).mean()
    return df

# ------------------------------
# Backtesting
# ------------------------------
def backtest_signals(df: pd.DataFrame, initial_capital: float = 10000, leverage: float = 1.0, lot_size: float = 1.0,
                     rr_target: float = 2.0, forced_exit_rr: float = 5.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size
    df["capital"] = initial_capital * (1 + df["strategy_returns"].cumsum())
    df["drawdown"] = df["capital"] / df["capital"].cummax() - 1
    max_rr = df["strategy_returns"].max() * 100
    achieved_rr = df["strategy_returns"].sum() * 100
    metrics = {
        "max_rr": max_rr,
        "achieved_rr": achieved_rr,
        "final_capital": df["capital"].iloc[-1],
        "drawdown": df["drawdown"].min()
    }
    # Forced exit logic
    exit_idx = df[df["strategy_returns"].cumsum() >= forced_exit_rr / 100].index
    if not exit_idx.empty:
        df.loc[exit_idx[0]:, "position"] = 0
    return df, metrics

# ------------------------------
# Streamlit Sidebar Inputs
# ------------------------------
st.sidebar.title("Backtester Parameters")
selected_asset_name = st.sidebar.selectbox(
    "Select Asset", options=[a.name for a in assets_list]
)
selected_asset = next((a for a in assets_list if a.name == selected_asset_name), None)

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0, step=1000.0)
leverage = st.sidebar.number_input("Leverage", value=1.0, step=0.1)
lot_size = st.sidebar.number_input("Lot Size", value=1.0, step=0.1)
buy_threshold = st.sidebar.slider("Buy Threshold", 0.0, 0.1, 0.01)
sell_threshold = st.sidebar.slider("Sell Threshold", 0.0, 0.1, 0.01)
atr_lookback = st.sidebar.slider("ATR Lookback", 5, 50, 14)
rvol_lookback = st.sidebar.slider("RVol Lookback", 5, 50, 14)
rr_target = st.sidebar.number_input("1R (%)", value=2.0, step=0.1)
forced_exit_rr = st.sidebar.number_input("Forced Exit R (%)", value=5.0, step=0.1)


# ===== Chunk 3/3 =====
def main():
    if selected_asset is None:
        st.error("Selected asset not found in asset list.")
        return

    # Fetch data
    cot_df, price_df = fetch_all_data(selected_asset, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if price_df is None or price_df.empty:
        st.error("Price data not available for the selected asset.")
        return
    if cot_df is None or cot_df.empty:
        st.warning("COT data not available. Health gauge will be zero.")
        cot_df = pd.DataFrame()

    # Health gauge
    health = calculate_health_gauge(cot_df)
    st.metric("Health Gauge", f"{health:.2f}")

    # Merge data for signals
    merged = price_df.copy()
    # Add placeholder for COT if needed
    if not cot_df.empty:
        merged["commercial_net"] = cot_df["commercial_net"]

    # Generate signals
    signals_df = generate_signals(
        merged,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        atr_lookback=atr_lookback,
        rvol_lookback=rvol_lookback
    )

    # Backtest
    backtest_df, metrics = backtest_signals(
        signals_df,
        initial_capital=initial_capital,
        leverage=leverage,
        lot_size=lot_size,
        rr_target=rr_target,
        forced_exit_rr=forced_exit_rr
    )

    # Display metrics
    st.subheader("Backtest Metrics")
    st.write(metrics)

    # Plot equity curve
    st.subheader("Equity Curve")
    st.line_chart(backtest_df["capital"])

    # Plot positions
    st.subheader("Positions")
    st.line_chart(backtest_df["position"])

if __name__ == "__main__":
    main()
