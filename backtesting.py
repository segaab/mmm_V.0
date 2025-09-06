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

# --- Constants and Setup ---
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
                # derive nets safely
                df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
                df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
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
                hist = hist.loc[ticker].reset_index()
            else:
                hist = hist.reset_index().rename(columns={"index": "date"})
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
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
    if df is None or df.empty:
        return pd.Series()
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback, min_periods=1).mean()
    return atr

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

# --- Health gauge calculation ---
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
        oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0
    except Exception:
        oi_score = 0.0

    # COT analytics (35%)
    try:
        short_term = cot_df[cot_df["report_date"] >= three_months_ago]["commercial_net"].dropna()
        long_term = cot_df[cot_df["report_date"] >= one_year_ago]["non_commercial_net"].dropna()
        st_score = float((short_term.iloc[-1] - short_term.min()) / (short_term.max() - short_term.min() + 1e-9)) if not short_term.empty else 0.0
        lt_score = float((long_term.iloc[-1] - long_term.min()) / (long_term.max() - long_term.min() + 1e-9)) if not long_term.empty else 0.0
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

# --- Threaded batch fetch ---
def fetch_batch(batch_assets: List[Tuple[str, str]], start_date: pd.Timestamp, end_date: pd.Timestamp,
                cot_results: Dict[str, pd.DataFrame], price_results: Dict[str, pd.DataFrame], lock: threading.Lock):
    for cot_name, ticker in batch_assets:
        try:
            cot_df = fetch_cot_data(cot_name)
            if cot_df is None or cot_df.empty:
                with lock:
                    cot_results[cot_name] = pd.DataFrame()
                    price_results[cot_name] = pd.DataFrame()
                continue
            price_df = fetch_price_data_yahoo(ticker, start_date.isoformat(), end_date.isoformat())
            if price_df is None or price_df.empty:
                with lock:
                    cot_results[cot_name] = cot_df
                    price_results[cot_name] = pd.DataFrame()
                continue
            price_df = calculate_rvol(price_df)
            merged = merge_cot_price(cot_df, price_df)
            merged["health_score"] = calculate_health_gauge(cot_df, merged)
            with lock:
                cot_results[cot_name] = cot_df
                price_results[cot_name] = merged
        except Exception as e:
            logger.exception("Error loading data for %s: %s", cot_name, e)
            with lock:
                cot_results[cot_name] = pd.DataFrame()
                price_results[cot_name] = pd.DataFrame()

def fetch_all_data(assets_dict: Dict[str, str], start_date: pd.Timestamp, end_date: pd.Timestamp, batch_size: int = 5) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
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



# ------------------------------
# Backtesting & Signals
# ------------------------------

def calculate_price_bands(df: pd.DataFrame, window: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["ma"] = df["close"].rolling(window, min_periods=1).mean()
    df["std"] = df["close"].rolling(window, min_periods=1).std()
    df["upper_band"] = df["ma"] + std_mult * df["std"]
    df["lower_band"] = df["ma"] - std_mult * df["std"]
    return df

def generate_signals(df: pd.DataFrame, buy_threshold: float = 0.5, sell_threshold: float = 0.5,
                     rvol_threshold: float = 1.5, atr_lookback: int = 14, rvol_lookback: int = 20,
                     volatility_weight: float = 0.5, microstructure_weight: float = 0.5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = calculate_price_bands(df)
    df["atr"] = calculate_atr(df, atr_lookback)
    df["rvol"] = calculate_rvol(df, rvol_lookback)["rvol"]
    df["signal"] = 0
    # Entry logic: simplified weighted signal
    df.loc[(df["close"] > df["upper_band"]) & (df["rvol"] >= rvol_threshold), "signal"] = 1 * volatility_weight
    df.loc[(df["close"] < df["lower_band"]) & (df["rvol"] >= rvol_threshold), "signal"] = -1 * volatility_weight
    df["signal"] = df["signal"].round(0)
    return df

def backtest_signals(df: pd.DataFrame, signals_df: pd.DataFrame,
                     initial_capital: float = 10000, leverage: float = 1.0, lot_size: float = 1.0,
                     rr_target: float = 2.0, forced_exit_rr: float = 5.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df is None or df.empty or signals_df is None or signals_df.empty:
        return pd.DataFrame(), {}
    df = df.copy()
    df["position"] = signals_df["signal"].fillna(0)
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size

    # Track risk/reward per trade
    df["trade_entry"] = df["position"].diff().fillna(0)
    df["cum_returns"] = (1 + df["strategy_returns"]).cumprod() - 1
    df["max_rr"] = 0.0
    df["achieved_rr"] = 0.0

    capital = initial_capital
    current_rr = 0.0
    for idx, row in df.iterrows():
        if row["trade_entry"] != 0:
            current_rr = 0.0  # reset RR on new trade
        current_rr += row["strategy_returns"]
        df.at[idx, "achieved_rr"] = current_rr
        df.at[idx, "max_rr"] = max(df.at[idx, "max_rr"], abs(current_rr))
        # Forced exit logic
        if abs(current_rr) >= forced_exit_rr * rr_target:
            df.at[idx, "position"] = 0

    metrics = {
        "total_return": df["strategy_returns"].sum(),
        "final_capital": initial_capital + df["strategy_returns"].sum() * initial_capital,
        "max_rr": df["max_rr"].max(),
        "achieved_rr": df["achieved_rr"].max(),
        "sharpe": df["strategy_returns"].mean() / (df["strategy_returns"].std() + 1e-9) * np.sqrt(252)
    }
    return df, metrics

# ------------------------------
# Main execution
# ------------------------------

def main():
    st.set_page_config(page_title="Backtester Dashboard", layout="wide")

    # Sidebar controls
    rr_target = st.sidebar.number_input("Risk per trade (%)", value=2.0, step=0.1)
    forced_exit_rr = st.sidebar.number_input("Forced exit multiplier (R)", value=5.0, step=0.5)
    initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0, step=100.0)
    leverage = st.sidebar.number_input("Leverage", value=1.0, step=0.1)
    lot_size = st.sidebar.number_input("Lot Size", value=1.0, step=0.1)
    atr_lookback = st.sidebar.number_input("ATR Lookback", value=14, step=1)
    rvol_lookback = st.sidebar.number_input("RVol Lookback", value=20, step=1)
    buy_threshold = st.sidebar.slider("Buy Signal Threshold", 0.0, 1.0, 0.5)
    sell_threshold = st.sidebar.slider("Sell Signal Threshold", 0.0, 1.0, 0.5)

    # Asset selection
    asset = st.sidebar.selectbox("Select Asset", options=[a.symbol for a in assets_list])
    selected_obj = next((a for a in assets_list if a.symbol == asset), None)
    if selected_obj is None:
        st.error("Selected asset object not found.")
        return

    # Fetch data
    start_date_pd = pd.Timestamp(selected_obj.start_date)
    end_date_pd = pd.Timestamp(selected_obj.end_date)
    cot_results, price_results = fetch_all_data({selected_obj.cot_name: selected_obj.ticker}, start_date_pd, end_date_pd)
    price_df = price_results[selected_obj.cot_name]

    # Generate signals
    signals_df = generate_signals(price_df, buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                  atr_lookback=atr_lookback, rvol_lookback=rvol_lookback)

    # Backtest
    backtest_df, metrics = backtest_signals(price_df, signals_df, initial_capital=initial_capital,
                                            leverage=leverage, lot_size=lot_size, rr_target=rr_target,
                                            forced_exit_rr=forced_exit_rr)

    # Display
    st.subheader("Backtest Metrics")
    st.write(metrics)

    st.subheader("Backtest Chart")
    st.line_chart(backtest_df[["cum_returns"]])

if __name__ == "__main__":
    main()
