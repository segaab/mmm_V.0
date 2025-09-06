# ===== Chunk 1/3 =====
import os
import logging
import time
import threading
from typing import Dict, Tuple, List
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

# --- Asset Mapping ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "DOW JONES INDUSTRIAL AVERAGE - CHICAGO MERCANTILE EXCHANGE": "YM=F",
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
}

# ------------------------------
# Data fetching helpers
# ------------------------------
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
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
            # Normalize column names
            hist.columns = [c.lower() for c in hist.columns]
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
    """Calculate ATR for volatility."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=1).mean()
    return atr

def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly COT onto daily prices using merge_asof."""
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

def calculate_price_bands(price_df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Add upper, lower, mid bands, range, and extension metrics."""
    if price_df is None or price_df.empty:
        return price_df
    df = price_df.copy()
    close_col = "close"
    high_col = "high"
    low_col = "low"

    df["upper_band"] = df[high_col].rolling(lookback, min_periods=1).max()
    df["lower_band"] = df[low_col].rolling(lookback, min_periods=1).min()
    df["mid_band"] = (df["upper_band"] + df["lower_band"]) / 2.0
    df["range"] = df["upper_band"] - df["lower_band"]
    df["extension"] = ((df[close_col] - df["mid_band"]) / (df["range"] / 2 + 1e-9)).fillna(0.0)
    return df

def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    """Compute composite health score combining open interest, COT, and price signals."""
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
        if oi_series.empty:
            oi_score = 0.0
        else:
            oi_norm = (oi_series - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)
            oi_score = float(oi_norm.iloc[-1])
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
            recent["return"] = recent["close"].pct_change().fillna(0.0)
            rvol_75 = recent["rvol"].quantile(0.75)
            recent["vol_avg20"] = recent["volume"].rolling(20).mean()
            recent["vol_spike"] = recent["volume"] > recent["vol_avg20"]
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
# Backtesting
# ------------------------------

def backtest_signals(df: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 10000,
                     leverage: float = 1.0, lot_size: float = 1.0, rr_per_trade: float = 0.02,
                     max_rr: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Run backtest with forced exit at max R:R."""
    if df is None or df.empty or signals is None or signals.empty:
        return pd.DataFrame(), {}

    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df = df.merge(signals[["date", "signal"]], on="date", how="left")
    df["position"] = df["signal"].ffill().fillna(0)

    # Track per-trade R:R
    trade_entry_price = None
    rr_tracker = []
    positions = []

    for i, row in df.iterrows():
        pos = row["position"]
        close = row["close"]

        if pos != 0 and trade_entry_price is None:
            trade_entry_price = close
            trade_rr = 0.0
        elif pos != 0 and trade_entry_price is not None:
            trade_rr = abs(close - trade_entry_price) / (rr_per_trade * trade_entry_price)
            if trade_rr >= max_rr:
                pos = 0
                trade_entry_price = None
        else:
            trade_entry_price = None
            trade_rr = 0.0
        rr_tracker.append(trade_rr)
        positions.append(pos)

    df["position"] = positions
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size
    df["cum_returns"] = (1 + df["strategy_returns"]).cumprod() * initial_capital

    metrics = {
        "total_return": df["cum_returns"].iloc[-1] - initial_capital,
        "max_rr": max_rr,
        "avg_rr": np.mean(rr_tracker)
    }
    return df, metrics

# ------------------------------
# Main
# ------------------------------

def main():
    st.title("Market Backtester")
    st.sidebar.header("Backtest Parameters")

    # Sidebar controls
    asset = st.sidebar.selectbox("Select Asset", options=[a.symbol for a in assets_list])
    initial_capital = st.sidebar.number_input("Starting Capital", value=10000.0, step=1000.0)
    leverage = st.sidebar.number_input("Leverage", value=1.0, step=0.1)
    lot_size = st.sidebar.number_input("Lot Size", value=1.0, step=0.1)
    rr_per_trade = st.sidebar.slider("R:R per trade", min_value=0.005, max_value=0.1, value=0.02, step=0.005)
    max_rr = st.sidebar.slider("Max R:R Forced Exit", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    atr_lookback = st.sidebar.number_input("ATR Lookback", value=14, step=1)
    rvol_lookback = st.sidebar.number_input("RVol Lookback", value=20, step=1)
    buy_threshold = st.sidebar.slider("Buy Signal Threshold", 0.0, 1.0, 0.5, 0.01)
    sell_threshold = st.sidebar.slider("Sell Signal Threshold", 0.0, 1.0, 0.5, 0.01)

    # Fetch data
    start_date_pd = pd.to_datetime("2023-01-01")
    end_date_pd = pd.to_datetime("2025-01-01")
    cot_results, price_results = fetch_all_data([asset], start_date_pd, end_date_pd)
    merged = merge_cot_price(cot_results[asset], price_results[asset])
    merged = calculate_price_bands(merged)
    merged = calculate_rvol(merged, rvol_lookback)
    merged["atr"] = calculate_atr(merged, atr_lookback)

    # Generate signals
    signals_df = generate_signals(merged, buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                  rvol_threshold=1.5, atr_lookback=atr_lookback, rvol_lookback=rvol_lookback)

    # Calculate health
    health_score = calculate_health_gauge(cot_results[asset], merged)
    st.metric("Health Gauge", f"{health_score:.2f}/10")

    # Backtest
    backtest_df, metrics = backtest_signals(merged, signals_df, initial_capital=initial_capital,
                                            leverage=leverage, lot_size=lot_size,
                                            rr_per_trade=rr_per_trade, max_rr=max_rr)

    st.subheader("Backtest Metrics")
    st.json(metrics)

    st.subheader("Equity Curve")
    st.line_chart(backtest_df.set_index("date")["cum_returns"])

if __name__ == "__main__":
    main()
