import os
import logging
import time
import threading
from typing import Dict, Tuple, List
from datetime import datetime, timedelta, date

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
    df = df.copy()
    high = df["high"] if "high" in df.columns else df["High"]
    low = df["low"] if "low" in df.columns else df["Low"]
    close = df["close"] if "close" in df.columns else df["Close"]
    df["tr1"] = high - low
    df["tr2"] = (high - close.shift(1)).abs()
    df["tr3"] = (low - close.shift(1)).abs()
    tr = df[["tr1", "tr2", "tr3"]].max(axis=1)
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

        st_score = 0.0 if short_term.empty else float(
            (short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) /
            (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9)
        )
        lt_score = 0.0 if long_term.empty else float(
            (long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) /
            (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9)
        )

        cot_analytics_score = 0.4 * st_score + 0.6 * lt_score
    except Exception:
        cot_analytics_score = 0.0

    # Price + RVol score (40%)
    try:
        recent = price_df[price_df["date"] >= three_months_ago].copy()
        if recent.empty or "rvol" not in recent.columns:
            pv_score = 0.0
        else:
            close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
            vol_col = "volume" if "volume" in recent.columns else ("Volume" if "Volume" in recent.columns else None)
            if close_col is None or vol_col is None:
                pv_score = 0.0
            else:
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
# Signal generation with ATR and RVol
# ------------------------------

def generate_signals(price_df: pd.DataFrame, buy_threshold: float = 7.0, sell_threshold: float = 3.0,
                     rvol_threshold: float = 1.5, atr_lookback: int = 14, rvol_lookback: int = 20) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    df = price_df.copy()

    df = calculate_rvol(df, window=rvol_lookback)
    df["atr"] = calculate_atr(df, lookback=atr_lookback)

    if "health_score" not in df.columns:
        df["health_score"] = 5.0

    df = calculate_price_bands(df)

    # Entry logic
    df["buy_signal"] = (df["health_score"] >= buy_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] < 0)
    df["sell_signal"] = (df["health_score"] <= sell_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] > 0)
    df["strong_buy"] = df["buy_signal"] & (df["rvol"] > 2.0) & (df["extension"] < -0.5)
    df["strong_sell"] = df["sell_signal"] & (df["rvol"] > 2.0) & (df["extension"] > 0.5)

    return df

# ------------------------------
# Backtesting with R:R and forced exit
# ------------------------------

def backtest_signals(price_df: pd.DataFrame, signals_df: pd.DataFrame,
                     initial_capital: float = 100000, leverage: float = 1.0,
                     lot_size: float = 1.0, rr_per_trade: float = 0.02, forced_exit_rr: float = 0.10):
    df = price_df.copy()
    df = pd.merge(df, signals_df[["date", "buy_signal", "sell_signal", "strong_buy", "strong_sell"]], on="date", how="left")
    df.fillna(False, inplace=True)

    df["position"] = 0
    entry_price = None
    max_rr_achieved = 0.0
    rr_list = []

    for i in range(1, len(df)):
        # Entry
        if df.loc[i, "buy_signal"]:
            df.loc[i, "position"] = 1
            entry_price = df.loc[i, "close"]
        elif df.loc[i, "sell_signal"]:
            df.loc[i, "position"] = -1
            entry_price = df.loc[i, "close"]
        else:
            df.loc[i, "position"] = df.loc[i - 1, "position"]

        # Exit based on forced RR
        if entry_price is not None:
            move = (df.loc[i, "close"] - entry_price) / entry_price if df.loc[i, "position"] > 0 else (entry_price - df.loc[i, "close"]) / entry_price
            rr_list.append(move / rr_per_trade)
            max_rr_achieved = max(max_rr_achieved, move / rr_per_trade)
            if move >= forced_exit_rr:
                df.loc[i, "position"] = 0
                entry_price = None

    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size
    df["capital"] = initial_capital * (1 + df["strategy_returns"].cumsum())

    metrics = {
        "max_rr": max_rr_achieved,
        "average_rr": np.mean(rr_list) if rr_list else 0.0
    }

    return df, metrics

# ------------------------------
# Main function
# ------------------------------

def main():
    st.title("COT + Price Backtester")

    start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    buy_threshold = st.sidebar.slider("Buy Threshold", 0.0, 10.0, 7.0)
    sell_threshold = st.sidebar.slider("Sell Threshold", 0.0, 10.0, 3.0)
    rr_per_trade = st.sidebar.number_input("1R (%)", value=2.0, step=0.1) / 100.0
    forced_exit_rr = st.sidebar.number_input("Forced Exit R (%)", value=10.0, step=0.5) / 100.0
    leverage = st.sidebar.number_input("Leverage", value=1.0, step=0.1)
    lot_size = st.sidebar.number_input("Lot Size", value=1.0, step=0.1)
    atr_lookback = st.sidebar.number_input("ATR Lookback", value=14, step=1)
    rvol_lookback = st.sidebar.number_input("RVOL Lookback", value=20, step=1)

    selected_asset = st.sidebar.selectbox("Select Asset", list(assets.keys()))

    if st.sidebar.button("Run Backtest"):
        start_date_pd = pd.Timestamp(start_date)
        end_date_pd = pd.Timestamp(end_date)

        with st.spinner("Fetching data..."):
            cot_df = fetch_cot_data(selected_asset)
            price_df = fetch_price_data_yahoo(assets[selected_asset], start_date_pd.isoformat(), end_date_pd.isoformat())
            if price_df.empty or cot_df.empty:
                st.warning(f"No data for {selected_asset}")
                return
            price_df = calculate_rvol(price_df, window=rvol_lookback)
            merged = merge_cot_price(cot_df, price_df)
            merged["health_score"] = calculate_health_gauge(cot_df, merged)

        signals_df = generate_signals(merged, buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                      rvol_threshold=1.5, atr_lookback=atr_lookback, rvol_lookback=rvol_lookback)
        backtest_df, metrics = backtest_signals(merged, signals_df,
                                                initial_capital=100000, leverage=leverage,
                                                lot_size=lot_size, rr_per_trade=rr_per_trade,
                                                forced_exit_rr=forced_exit_rr)

        st.subheader(f"{selected_asset} Backtest Results")
        st.line_chart(backtest_df[["capital"]])
        st.write(backtest_df.tail(10))
        st.metric("Max R:R achieved", round(metrics["max_rr"], 2))
        st.metric("Average R:R per trade", round(metrics["average_rr"], 2))

if __name__ == "__main__":
    main()
