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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets Mapping ---
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
# Fetching Helpers
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
                df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", 0), errors="coerce")
                df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
                df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
                return df.sort_values("report_date").reset_index(drop=True)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error("Error fetching COT data for %s: %s", market_name, e)
            attempt += 1
            time.sleep(1 + attempt)
    return pd.DataFrame()

def fetch_price_data_yahoo(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info("Fetching Yahoo data for %s from %s to %s", ticker, start_date, end_date)
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist is None or hist.empty:
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker].reset_index()
            else:
                hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching Yahoo data for %s: %s", ticker, e)
            attempt += 1
            time.sleep(1 + attempt)
    return pd.DataFrame()


# ===== Chunk 2/3 =====

# ------------------------------
# Processing Helpers
# ------------------------------

def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
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

def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
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
    if price_df is None or price_df.empty:
        return price_df
    df = price_df.copy()
    close_col = "close" if "close" in df.columns else df.columns[0]
    high_col = "high" if "high" in df.columns else close_col
    low_col = "low" if "low" in df.columns else close_col

    df["upper_band"] = df[high_col].rolling(lookback, min_periods=1).max()
    df["lower_band"] = df[low_col].rolling(lookback, min_periods=1).min()
    df["mid_band"] = (df["upper_band"] + df["lower_band"]) / 2.0
    df["range"] = df["upper_band"] - df["lower_band"]
    df["extension"] = ((df[close_col] - df["mid_band"]) / (df["range"] / 2 + 1e-9)).fillna(0.0)
    return df

def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return float("nan")
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    last_date = price_df["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open interest score
    oi_series = price_df["open_interest_all"].dropna()
    oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0

    # COT analytics score
    commercial = cot_df[["report_date", "commercial_net"]].dropna()
    commercial["report_date"] = pd.to_datetime(commercial["report_date"], errors="coerce")
    short_term = commercial[commercial["report_date"] >= three_months_ago]
    noncomm = cot_df[["report_date", "non_commercial_net"]].dropna()
    noncomm["report_date"] = pd.to_datetime(noncomm["report_date"], errors="coerce")
    long_term = noncomm[noncomm["report_date"] >= one_year_ago]

    st_score = float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / 
                     (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9)) if not short_term.empty else 0.0
    lt_score = float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / 
                     (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9)) if not long_term.empty else 0.0
    cot_score = 0.4 * st_score + 0.6 * lt_score

    # Price + RVol score
    recent = price_df[price_df["date"] >= three_months_ago].copy()
    if recent.empty or "rvol" not in recent.columns:
        pv_score = 0.0
    else:
        recent["return"] = recent["close"].pct_change().fillna(0.0)
        rvol_75 = recent["rvol"].quantile(0.75)
        recent["vol_avg20"] = recent["volume"].rolling(20).mean()
        recent["vol_spike"] = recent["volume"] > recent["vol_avg20"]
        filt = recent[(recent["rvol"] >= rvol_75) & recent["vol_spike"]]
        if filt.empty:
            pv_score = 0.0
        else:
            last_ret = float(filt["return"].iloc[-1])
            bucket = 5 if last_ret >= 0.02 else 4 if last_ret >= 0.01 else 3 if last_ret >= -0.01 else 2 if last_ret >= -0.02 else 1
            pv_score = (bucket - 1) / 4.0

    return (0.25 * oi_score + 0.35 * cot_score + 0.40 * pv_score) * 10.0

# --- Threaded batch fetch ---
def fetch_batch(batch_assets: List[Tuple[str, str]], start_date: pd.Timestamp, end_date: pd.Timestamp,
                cot_results: Dict[str, pd.DataFrame], price_results: Dict[str, pd.DataFrame], lock: threading.Lock):
    for cot_name, ticker in batch_assets:
        try:
            cot_df = fetch_cot_data(cot_name)
            price_df = fetch_price_data_yahoo(ticker, start_date.isoformat(), end_date.isoformat())
            if price_df is not None and not price_df.empty:
                price_df = calculate_rvol(price_df)
                merged = merge_cot_price(cot_df, price_df)
                merged["health_score"] = calculate_health_gauge(cot_df, merged)
            else:
                merged = pd.DataFrame()
            with lock:
                cot_results[cot_name] = cot_df if cot_df is not None else pd.DataFrame()
                price_results[cot_name] = merged
        except Exception as e:
            logger.exception("Error loading data for %s: %s", cot_name, e)
            with lock:
                cot_results[cot_name] = pd.DataFrame()
                price_results[cot_name] = pd.DataFrame()

def fetch_all_data(assets_dict: Dict[str, str], start_date: pd.Timestamp, end_date: pd.Timestamp, batch_size: int = 5):
    cot_results: Dict[str, pd.DataFrame] = {}
    price_results: Dict[str, pd.DataFrame] = {}
    lock = threading.Lock()
    items = list(assets_dict.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    for i in range(0, len(batches), 2):
        threads: List[threading.Thread] = []
        for j in range(i, min(i + 2, len(batches))):
            t = threading.Thread(target=fetch_batch, args=(batches[j], start_date, end_date, cot_results, price_results, lock), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        time.sleep(0.5)

    return cot_results, price_results

# ===== Chunk 3/3 =====

# ------------------------------
# Backtesting & Signals
# ------------------------------

def generate_signals(df: pd.DataFrame, atr_lookback: int = 14, rvol_lookback: int = 20,
                     entry_weight: float = 0.5, reversal_weight: float = 0.5,
                     buy_threshold: float = 0.6, sell_threshold: float = 0.6) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0.0)

    # ATR-based volatility
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = np.abs(df["high"] - df["close"].shift(1))
    df["low_close"] = np.abs(df["low"] - df["close"].shift(1))
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(atr_lookback).mean()

    # RVol
    df = calculate_rvol(df, window=rvol_lookback)

    # Weighted signal
    df["signal_raw"] = entry_weight * df["extension"].fillna(0) + reversal_weight * df["rvol"].fillna(0)
    df["signal_raw"] = (df["signal_raw"] - df["signal_raw"].min()) / (df["signal_raw"].max() - df["signal_raw"].min() + 1e-9)

    df["position"] = 0
    df.loc[df["signal_raw"] >= buy_threshold, "position"] = 1
    df.loc[df["signal_raw"] <= 1 - sell_threshold, "position"] = -1
    df["position"] = df["position"].ffill().fillna(0)
    return df

def backtest_signals(df: pd.DataFrame, signals_df: pd.DataFrame, initial_capital: float = 100000,
                     leverage: float = 1.0, lot_size: float = 1.0,
                     rr_1r: float = 0.02, forced_exit_r: float = 5.0) -> Tuple[pd.DataFrame, Dict]:
    if df is None or df.empty or signals_df is None or signals_df.empty:
        return pd.DataFrame(), {}
    df = df.copy()
    df["position"] = signals_df["position"]
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size

    equity = [initial_capital]
    max_rr_list, achieved_rr_list = [], []
    current_r = 0.0

    for i in range(1, len(df)):
        trade_ret = df["strategy_returns"].iloc[i] * equity[-1]
        equity.append(equity[-1] + trade_ret)

        r_ratio = trade_ret / (initial_capital * rr_1r)
        current_r += r_ratio
        max_rr_list.append(current_r)
        achieved_rr_list.append(min(current_r, forced_exit_r))

        # Forced exit
        if current_r >= forced_exit_r:
            df.at[i, "position"] = 0
            current_r = 0.0

    df["equity"] = equity
    df["max_rr"] = [0] + max_rr_list
    df["achieved_rr"] = [0] + achieved_rr_list

    metrics = {
        "final_equity": equity[-1],
        "max_rr": max(max_rr_list) if max_rr_list else 0,
        "achieved_rr": max(achieved_rr_list) if achieved_rr_list else 0,
        "sharpe_ratio": np.mean(df["strategy_returns"]) / (np.std(df["strategy_returns"]) + 1e-9) * np.sqrt(252)
    }
    return df, metrics

# ------------------------------
# Streamlit Sidebar Controls
# ------------------------------

st.sidebar.header("Portfolio & Backtesting Controls")
initial_capital = st.sidebar.number_input("Starting Balance", value=100000.0, step=1000.0)
leverage = st.sidebar.slider("Leverage", 1.0, 10.0, 1.0, step=0.1)
lot_size = st.sidebar.slider("Lot Size", 0.1, 10.0, 1.0, step=0.1)
atr_lookback = st.sidebar.number_input("ATR Lookback", value=14, step=1)
rvol_lookback = st.sidebar.number_input("RVol Lookback", value=20, step=1)
entry_weight = st.sidebar.slider("Entry Weight", 0.0, 1.0, 0.5, step=0.05)
reversal_weight = st.sidebar.slider("Reversal Weight", 0.0, 1.0, 0.5, step=0.05)
buy_threshold = st.sidebar.slider("Buy Signal Threshold", 0.0, 1.0, 0.6, step=0.05)
sell_threshold = st.sidebar.slider("Sell Signal Threshold", 0.0, 1.0, 0.6, step=0.05)
rr_1r = st.sidebar.number_input("1R (%)", value=2.0, step=0.1) / 100.0
forced_exit_r = st.sidebar.number_input("Forced Exit (R)", value=5.0, step=0.1)

# ------------------------------
# Main Function
# ------------------------------

def main():
    assets_dict = {
        "COT_OIL": "CL=F",
        "COT_GOLD": "GC=F"
    }

    start_date_pd = pd.Timestamp("2023-01-01")
    end_date_pd = pd.Timestamp("2025-01-01")

    cot_results, price_results = fetch_all_data(assets_dict, start_date_pd, end_date_pd)

    for asset, df in price_results.items():
        signals_df = generate_signals(df, atr_lookback, rvol_lookback, entry_weight, reversal_weight, buy_threshold, sell_threshold)
        backtest_df, metrics = backtest_signals(df, signals_df, initial_capital, leverage, lot_size, rr_1r, forced_exit_r)
        st.write(f"Asset: {asset}")
        st.line_chart(backtest_df[["equity"]])
        st.write(metrics)

if __name__ == "__main__":
    main()
