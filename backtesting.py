import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import matplotlib.pyplot as plt
from sodapy import Socrata
from yahooquery import Ticker
from datetime import timedelta

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page ---
st.set_page_config(page_title="Trading Strategy Backtester", page_icon="📈", layout="wide")

# --- COT API Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets Mapping ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
}

# --- Contract Sizes ---
CONTRACT_SIZES = {
    "FX": {"default": 100000},
    "OIL": {"WTI": 1000, "BRENT": 1000, "default": 1000},
    "METALS": {"XAUUSD": 100, "XAGUSD": 5000, "default": 100},
    "INDICES": {"SP500": 10, "DAX30": 25, "FTSE100": 10, "default": 10}
}

def get_contract_size(asset_class, symbol=None):
    asset_class = asset_class.upper()
    if asset_class not in CONTRACT_SIZES:
        raise ValueError(f"Unknown asset class: {asset_class}")

    if symbol and symbol.upper() in CONTRACT_SIZES[asset_class]:
        return CONTRACT_SIZES[asset_class][symbol.upper()]
    return CONTRACT_SIZES[asset_class]["default"]

# --- Asset Class Mapping ---
asset_classes = {
    "GOLD - COMMODITY EXCHANGE INC.": ("METALS", "XAUUSD"),
    "SILVER - COMMODITY EXCHANGE INC.": ("METALS", "XAGUSD"),
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": ("FX", "EURUSD"),
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": ("FX", "USDJPY"),
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": ("INDICES", "SP500"),
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": ("INDICES", "SP500"),
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": ("OIL", "WTI"),
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": ("OIL", "default"),
}

# --- Fetch COT data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500
            )
            if not results:
                return pd.DataFrame()
            df = pd.DataFrame.from_records(results)
            df["report_date"] = pd.to_datetime(df.get("report_date_as_yyyy_mm_dd", pd.NaT), errors="coerce")
            df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
            df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce") - pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
            df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", 0), errors="coerce")
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching COT for %s: %s", market_name, e)
            attempt += 1
    logger.error("Failed to fetch COT for %s after %d attempts.", market_name, max_attempts)
    return pd.DataFrame()

# --- Fetch price data ---
def fetch_price_data_yahoo(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist is None or hist.empty:
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                try:
                    hist = hist.loc[ticker]
                except:
                    hist = hist.reset_index(level=0, drop=True)
            hist = hist.reset_index()
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            else:
                hist.index = pd.to_datetime(hist.index)
                hist = hist.reset_index().rename(columns={"index": "date"})
            hist["close"] = pd.to_numeric(hist.get("close", np.nan), errors="coerce")
            hist["volume"] = pd.to_numeric(hist.get("volume", np.nan), errors="coerce")
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching Yahoo data for %s: %s", ticker, e)
            attempt += 1
    logger.error("Failed fetching Yahoo data for %s after %d attempts.", ticker, max_attempts)
    return pd.DataFrame()

# --- Calculate Relative Volume (RVol) ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if df is None or df.empty or "volume" not in df.columns:
        df["rvol"] = np.nan
        return df
    df = df.copy()
    df["rvol"] = df["volume"] / df["volume"].rolling(window, min_periods=1).mean()
    return df

# --- Merge COT + Price ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()
    for col in ["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]:
        if col not in cot_df.columns:
            cot_df[col] = np.nan
    cot_small = cot_df[["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]].copy()
    cot_small.rename(columns={"report_date": "date"}, inplace=True)
    price_df["date"] = pd.to_datetime(price_df["date"])
    merged = pd.merge_asof(price_df.sort_values("date"),
                           cot_small.sort_values("date"),
                           on="date",
                           direction="backward")
    for col in ["open_interest_all", "commercial_net", "non_commercial_net"]:
        merged[col] = merged[col].ffill()
    return merged

# --- Calculate Health Gauge ---
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return float("nan")
    df = price_df.copy()
    df["rvol"] = df.get("rvol", np.nan)
    last_date = df["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open Interest score (25%)
    oi_series = df["open_interest_all"].dropna()
    oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0

    # COT analytics (35%)
    commercial = cot_df[["report_date", "commercial_net"]].dropna(subset=["commercial_net"])
    non_commercial = cot_df[["report_date", "non_commercial_net"]].dropna(subset=["non_commercial_net"])
    short_term = commercial[commercial["report_date"] >= three_months_ago]
    long_term = non_commercial[non_commercial["report_date"] >= one_year_ago]
    st_score = float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9)) if not short_term.empty else 0.0
    lt_score = float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9)) if not long_term.empty else 0.0
    cot_score = 0.4 * st_score + 0.6 * lt_score

    # Price + RVol score (40%)
    recent = df[df["date"] >= three_months_ago]
    if recent.empty or "rvol" not in recent.columns or recent["rvol"].isna().all():
        pv_score = 0.0
    else:
        rvol_75 = recent["rvol"].quantile(0.75)
        recent["vol_avg20"] = recent["volume"].rolling(20, min_periods=1).mean()
        recent["vol_spike"] = recent["volume"] > recent["vol_avg20"]
        filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
        if filt.empty:
            pv_score = 0.0
        else:
            last_ret = float(filt["close"].pct_change().iloc[-1]) if len(filt) > 1 else 0.0
            bucket = 5 if last_ret >= 0.02 else 4 if last_ret >= 0.01 else 3 if last_ret >= -0.01 else 2 if last_ret >= -0.02 else 1
            pv_score = (bucket - 1) / 4.0

    return (0.25 * oi_score + 0.35 * cot_score + 0.4 * pv_score) * 10.0

# --- Generate signals ---
def generate_signals(df, buy_threshold=0.3, sell_threshold=0.7):
    if df.empty:
        return df

    health_gauges = []
    for i in range(len(df)):
        date = df.iloc[i]["date"]
        cot_subset = df[df["date"] <= date].copy()
        price_subset = df[df["date"] <= date].copy()
        hg = calculate_health_gauge(cot_subset, price_subset) if not cot_subset.empty and not price_subset.empty else np.nan
        health_gauges.append(hg)

    df["hg"] = health_gauges
    df["hg"] = df["hg"].fillna(0).clip(0, 10) / 10

    df["signal"] = 0
    df.loc[df["hg"] > sell_threshold, "signal"] = -1
    df.loc[df["hg"] < buy_threshold, "signal"] = 1

    return df