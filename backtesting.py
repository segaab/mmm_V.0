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

# --- Assets Mapping (without RBOB Gasoline and Heating Oil) ---
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

def calculate_price_bands(price_df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Calculate upper, lower, mid bands and extension for price."""
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
    """Calculate a combined health score based on COT, price, and volume analytics."""
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
    """Fetch all assets in batches with threading."""
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
# Signal generation
# ------------------------------

def generate_signals(price_df: pd.DataFrame, buy_threshold: float, sell_threshold: float, rvol_threshold: float = 1.5) -> pd.DataFrame:
    """Generate buy/sell signals based on health gauge, RVOL, and extension."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    df = price_df.copy()
    if "health_score" not in df.columns:
        df["health_score"] = 5.0

    df = calculate_price_bands(df)
    df = calculate_rvol(df)

    df["buy_signal"] = (df["health_score"] >= buy_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] < 0)
    df["sell_signal"] = (df["health_score"] <= sell_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] > 0)
    df["strong_buy"] = df["buy_signal"] & (df["rvol"] > 2.0) & (df["extension"] < -0.5)
    df["strong_sell"] = df["sell_signal"] & (df["rvol"] > 2.0) & (df["extension"] > 0.5)
    return df

# ------------------------------
# Exit Timing Neural Network
# ------------------------------

class ExitTimingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.layer3(x)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()
    df["rolling_mean_5"] = df["close"].rolling(5).mean()
    df["rolling_mean_20"] = df["close"].rolling(20).mean()
    df["rolling_std_5"] = df["close"].rolling(5).std()
    df["rolling_std_20"] = df["close"].rolling(20).std()
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    if "commercial_net" in df.columns:
        df["comm_net_change"] = df["commercial_net"].diff()
    if "non_commercial_net" in df.columns:
        df["non_comm_net_change"] = df["non_commercial_net"].diff()
    if "extension" in df.columns:
        df["extension_abs"] = df["extension"].abs()
    if "health_score" in df.columns:
        df["health_change"] = df["health_score"].diff()
    df.dropna(inplace=True)
    return df

def train_exit_model(df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    df = prepare_features(df.copy())
    if df.empty or len(df) < 50:
        raise ValueError("Not enough data to train exit model")

    df["future_returns_5d"] = df["close"].pct_change(5).shift(-5)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("No data after computing future returns")

    feature_cols = ["returns", "log_returns", "rolling_mean_5", "rolling_mean_20",
                    "rolling_std_5", "rolling_std_20", "momentum_5", "momentum_20"]
    for c in ["commercial_net", "non_commercial_net", "comm_net_change", "non_comm_net_change", "health_score", "extension"]:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].values
    y = df["future_returns_5d"].values.reshape(-1, 1)

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ExitTimingModel(input_size=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d completed", epoch + 1, epochs)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        test_loss = float(((preds - y_test) ** 2).mean())
        logger.info("Exit model test MSE: %.6f", test_loss)

    return model, scaler_X, scaler_y

# ------------------------------
# Backtesting with leverage & R:R
# ------------------------------

def backtest_signals(price_df: pd.DataFrame, signals_df: pd.DataFrame, starting_balance: float, leverage: float, lot_size: int):
    df = price_df.copy()
    df = pd.merge(df, signals_df[["date", "buy_signal", "sell_signal", "strong_buy", "strong_sell"]], on="date", how="left")
    df.fillna(False, inplace=True)

    df["position"] = 0
    for i in range(1, len(df)):
        if df.loc[i, "buy_signal"]:
            df.loc[i, "position"] = 1
        elif df.loc[i, "sell_signal"]:
            df.loc[i, "position"] = -1
        else:
            df.loc[i, "position"] = df.loc[i - 1, "position"]

    df["strategy_returns"] = df["returns"] * df["position"].shift(1) * leverage * lot_size
    df["capital"] = starting_balance * (1 + df["strategy_returns"].cumsum())

    # Risk/Reward tracking
    df["rr_ratio"] = np.nan
    max_rr = 0
    achieved_rr = 0
    for i in range(len(df)):
        if df.loc[i, "position"] != 0:
            entry_price = df["close"].iloc[i]
            potential_exit = df["close"].iloc[i:]  # simplistic future price
            if df.loc[i, "position"] == 1:
                rr = (potential_exit.max() - entry_price) / (entry_price - potential_exit.min() + 1e-9)
            else:
                rr = (entry_price - potential_exit.min()) / (potential_exit.max() - entry_price + 1e-9)
            df.at[i, "rr_ratio"] = rr
            if rr > max_rr:
                max_rr = rr
            if rr > achieved_rr:
                achieved_rr = rr

    return df, max_rr, achieved_rr

# ------------------------------
# Streamlit Main Function
# ------------------------------

def main():
    st.title("COT + Price Backtester")

    start_date = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    buy_threshold = st.sidebar.slider("Buy Threshold", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
    sell_threshold = st.sidebar.slider("Sell Threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    starting_balance = st.sidebar.number_input("Starting Balance", value=100000.0, step=1000.0)
    leverage = st.sidebar.number_input("Leverage", value=1.0, step=0.1)
    lot_size = st.sidebar.number_input("Lot Size", value=1, step=1)
    exit_days = st.sidebar.slider("Exit Horizon (Days)", min_value=1, max_value=20, value=5)

    selected_assets = st.sidebar.multiselect("Select Assets", list(assets.keys()), default=list(assets.keys())[:3])

    if st.sidebar.button("Run Backtest"):
        start_date_pd = pd.Timestamp(start_date)
        end_date_pd = pd.Timestamp(end_date)
        assets_dict = {k: assets[k] for k in selected_assets}

        with st.spinner("Fetching data..."):
            cot_results, price_results = fetch_all_data(assets_dict, start_date_pd, end_date_pd)

        for name in selected_assets:
            price_df = price_results.get(name)
            if price_df is None or price_df.empty:
                st.warning(f"No price data for {name}")

if __name__ == "__main__":
    main()
