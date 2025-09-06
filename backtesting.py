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

# --- Asset object list ---
class Asset:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol

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


# ------------------------------
# Price data fetching and processing
# ------------------------------

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
            # if MultiIndex (ticker, date), select ticker
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


# ------------------------------
# Processing helpers
# ------------------------------

def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rvol column to df (volume / rolling mean volume)."""
    if df is None or df.empty:
        return df
    df = df.copy()
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        df["rvol"] = np.nan
        return df
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    df["rvol"] = df[vol_col] / df[vol_col].rolling(window, min_periods=1).mean()
    return df


def calculate_atr(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Calculate ATR for volatility-based entry/exit."""
    high = df["high"] if "high" in df.columns else df["High"]
    low = df["low"] if "low" in df.columns else df["Low"]
    close = df["close"] if "close" in df.columns else df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1)
    atr = tr.max(axis=1).rolling(lookback, min_periods=1).mean()
    return atr


def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly COT onto daily prices using merge_asof. Forward-fill COT-derived fields."""
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return pd.DataFrame()
    cot_small = cot_df[["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]].copy()
    cot_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_small["date"] = pd.to_datetime(cot_small["date"], errors="coerce")

    price = price_df.copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.sort_values("date")
    cot_small = cot_small.sort_values("date")

    full_dates = pd.DataFrame({"date": pd.date_range(price["date"].min(), price["date"].max())})
    cot_on_dates = pd.merge_asof(full_dates, cot_small, on="date", direction="backward")

    merged = pd.merge(price, cot_on_dates[["date", "open_interest_all", "commercial_net", "non_commercial_net"]], on="date", how="left")
    merged["open_interest_all"] = merged["open_interest_all"].ffill()
    merged["commercial_net"] = merged["commercial_net"].ffill()
    merged["non_commercial_net"] = merged["non_commercial_net"].ffill()
    return merged


# ------------------------------
# Health gauge calculation
# ------------------------------

def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return float("nan")
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    last_date = price_df["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open interest score (25%)
    try:
        oi_series = price_df["open_interest_all"].dropna()
        oi_score = 0.0 if oi_series.empty else float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9))
    except Exception:
        oi_score = 0.0

    # COT analytics (35%)
    try:
        commercial = cot_df[["report_date", "commercial_net"]].dropna().copy()
        commercial["report_date"] = pd.to_datetime(commercial["report_date"], errors="coerce")
        short_term = commercial[commercial["report_date"] >= three_months_ago]

        noncomm = cot_df[["report_date", "non_commercial_net"]].dropna().copy()
        noncomm["report_date"] = pd.to_datetime(noncomm["report_date"], errors="coerce")
        long_term = noncomm[noncomm["report_date"] >= one_year_ago]

        st_score = 0.0 if short_term.empty else float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9))
        lt_score = 0.0 if long_term.empty else float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9))

        cot_analytics_score = 0.4 * st_score + 0.6 * lt_score
    except Exception:
        cot_analytics_score = 0.0

    # Price + RVol score (40%)
    try:
        recent = price_df[price_df["date"] >= three_months_ago].copy()
        if recent.empty or "rvol" not in recent.columns:
            pv_score = 0.0
        else:
            close_col = "close" if "close" in recent.columns else "Close"
            vol_col = "volume" if "volume" in recent.columns else "Volume"
            recent["return"] = recent[close_col].pct_change().fillna(0.0)
            rvol_75 = recent["rvol"].quantile(0.75)
            recent["vol_avg20"] = recent[vol_col].rolling(20).mean()
            recent["vol_spike"] = recent[vol_col] > recent["vol_avg20"]
            filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
            if filt.empty:
                pv_score = 0.0
            else:
                last_ret = float(filt["return"].iloc[-1])
                bucket = 5 if last_ret >= 0.02 else 4 if last_ret >= 0.01 else 3 if last_ret >= -0.01 else 2 if last_ret >= -0.02 else 1
                pv_score = (bucket - 1) / 4.0
    except Exception:
        pv_score = 0.0

    return (0.25 * oi_score + 0.35 * cot_analytics_score + 0.40 * pv_score) * 10.0

# ------------------------------
# Signal generation and backtesting
# ------------------------------

def calculate_price_bands(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    """Compute rolling mean and Bollinger bands."""
    df = df.copy()
    close_col = "close" if "close" in df.columns else "Close"
    df["ma"] = df[close_col].rolling(window, min_periods=1).mean()
    df["std"] = df[close_col].rolling(window, min_periods=1).std()
    df["upper"] = df["ma"] + std_mult * df["std"]
    df["lower"] = df["ma"] - std_mult * df["std"]
    return df


def generate_signals(df: pd.DataFrame, buy_threshold: float = 0.5, sell_threshold: float = 0.5,
                     atr_lookback: int = 14, rvol_lookback: int = 20, rvol_threshold: float = 1.5) -> pd.DataFrame:
    """Generate entry and exit signals based on price bands, ATR, and RVol."""
    df = df.copy()
    df = calculate_price_bands(df)
    df = calculate_rvol(df, rvol_lookback)
    df["atr"] = calculate_atr(df, atr_lookback)

    # Entry signals
    df["buy_signal"] = ((df["close"] < df["lower"]) & (df["rvol"] > rvol_threshold)).astype(int)
    df["sell_signal"] = ((df["close"] > df["upper"]) & (df["rvol"] > rvol_threshold)).astype(int)

    # Apply adjustable thresholds
    df["position"] = 0
    df.loc[df["buy_signal"] >= buy_threshold, "position"] = 1
    df.loc[df["sell_signal"] >= sell_threshold, "position"] = -1

    return df


def backtest_signals(df: pd.DataFrame, rr_target: float = 2.0, forced_exit_rr: float = 5.0,
                     initial_capital: float = 10000, leverage: float = 1.0, lot_size: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Backtest strategy with R:R target and forced exit."""
    df = df.copy()
    close_col = "close" if "close" in df.columns else "Close"
    df["returns"] = df[close_col].pct_change().fillna(0.0)
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size

    # Risk/Reward tracking
    max_rr = 0.0
    achieved_rr = 0.0
    open_trade_idx = None
    entry_price = None

    for i in range(len(df)):
        pos = df.at[i, "position"]
        price = df.at[i, close_col]

        if pos != 0 and open_trade_idx is None:
            # New trade
            open_trade_idx = i
            entry_price = price
            max_rr = 0.0

        if open_trade_idx is not None:
            # Update RR
            rr = (price - entry_price) / entry_price if df.at[open_trade_idx, "position"] > 0 else (entry_price - price) / entry_price
            max_rr = max(max_rr, rr)
            achieved_rr = rr

            # Forced exit
            if rr >= forced_exit_rr / 100:  # convert % to decimal
                df.at[i, "position"] = 0
                open_trade_idx = None
                entry_price = None

            # Exit on RR target
            elif rr >= rr_target / 100:
                df.at[i, "position"] = 0
                open_trade_idx = None
                entry_price = None

    metrics = {
        "cumulative_returns": float(df["strategy_returns"].cumsum().iloc[-1]),
        "max_rr": max_rr,
        "achieved_rr": achieved_rr,
        "total_trades": int((df["position"].diff().abs() > 0).sum() / 2)
    }
    return df, metrics


# ------------------------------
# Main Streamlit dashboard
# ------------------------------

def main():
    st.set_page_config(page_title="Backtester Dashboard", layout="wide")
    st.title("Market Backtester with Health Gauge")

    # Sidebar inputs
    asset = st.sidebar.selectbox("Select Asset", options=[a.symbol for a in assets_list])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

    initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0)
    leverage = st.sidebar.number_input("Leverage", value=1.0)
    lot_size = st.sidebar.number_input("Lot Size", value=1.0)
    buy_threshold = st.sidebar.slider("Buy Signal Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sell_threshold = st.sidebar.slider("Sell Signal Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    rr_target = st.sidebar.number_input("R:R Target (%)", value=2.0)
    forced_exit_rr = st.sidebar.number_input("Forced Exit RR (%)", value=5.0)

    # Fetch and merge data
    price_df = fetch_price_data_yahoo(asset, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    cot_df = fetch_cot_data(asset)  # ensure this function exists
    merged = merge_cot_price(cot_df, price_df)

    # Calculate health gauge
    health_score = calculate_health_gauge(cot_df, price_df)
    st.metric("Health Gauge", round(health_score, 2))

    # Generate signals and backtest
    signals_df = generate_signals(merged, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    backtest_df, metrics = backtest_signals(signals_df, rr_target=rr_target, forced_exit_rr=forced_exit_rr,
                                            initial_capital=initial_capital, leverage=leverage, lot_size=lot_size)

    st.subheader("Backtest Metrics")
    st.json(metrics)
    st.subheader("Backtest Plot")
    st.line_chart(backtest_df["strategy_returns"].cumsum())

if __name__ == "__main__":
    main()
