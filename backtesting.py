
# ===== Chunk 1/3 =====
import os
import logging
import time
import threading
from typing import Dict, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
from sodapy import Socrata
from yahooquery import Ticker

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
    logger.info("Fetching COT data for %s", market_name)
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get("6dca-aqww", where=where_clause,
                                 order="report_date_as_yyyy_mm_dd DESC", limit=1500)
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
                df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", 0), errors="coerce")
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
    merged = pd.merge(price, cot_on_dates[["date", "open_interest_all", "commercial_net", "non_commercial_net"]],
                      on="date", how="left")
    merged["open_interest_all"] = merged["open_interest_all"].ffill()
    merged["commercial_net"] = merged["commercial_net"].ffill()
    merged["non_commercial_net"] = merged["non_commercial_net"].ffill()
    return merged

def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return float("nan")
    price_df = price_df.copy()
    last_date = price_df["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open interest
    try:
        oi_series = price_df["open_interest_all"].dropna()
        oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0
    except Exception:
        oi_score = 0.0

    # COT analytics
    try:
        short_term = cot_df[cot_df["report_date"] >= three_months_ago]
        long_term = cot_df[cot_df["report_date"] >= one_year_ago]
        st_score = float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / 
                         (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9)) if not short_term.empty else 0.0
        lt_score = float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / 
                         (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9)) if not long_term.empty else 0.0
        cot_analytics_score = 0.4 * st_score + 0.6 * lt_score
    except Exception:
        cot_analytics_score = 0.0

    # Price + RVOL
    try:
        recent = price_df[price_df["date"] >= three_months_ago].copy()
        if "rvol" not in recent.columns:
            recent = calculate_rvol(recent)
        recent["return"] = recent["close"].pct_change().fillna(0)
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
    except Exception:
        pv_score = 0.0

    return float((0.25 * oi_score + 0.35 * cot_analytics_score + 0.4 * pv_score) * 10.0)

# ------------------------------
# Threaded batching fetch
# ------------------------------

def fetch_batch(batch_assets: List[Tuple[str, str]], start_date: pd.Timestamp, end_date: pd.Timestamp,
                cot_results: Dict[str, pd.DataFrame], price_results: Dict[str, pd.DataFrame], lock: threading.Lock):
    for cot_name, ticker in batch_assets:
        try:
            cot_df = fetch_cot_data(cot_name)
            price_df = fetch_price_data_yahoo(ticker, start_date.isoformat(), end_date.isoformat())
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

def fetch_all_data(assets_dict: Dict[str, str], start_date: pd.Timestamp, end_date: pd.Timestamp, batch_size: int = 5):
    cot_results, price_results, lock = {}, {}, threading.Lock()
    items = list(assets_dict.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for i in range(0, len(batches), 2):
        threads = []
        for j in range(i, min(i + 2, len(batches))):
            t = threading.Thread(target=fetch_batch, args=(batches[j], start_date, end_date, cot_results, price_results, lock))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        time.sleep(0.5)
    return cot_results, price_results

# ===== Chunk 2/3 =====
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Price extension bands / price-based features ---
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

# --- Health gauge time series ---
def calculate_health_gauge_series(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return pd.DataFrame(columns=["date", "health_gauge"])

    price_df = price_df.copy().sort_values("date").reset_index(drop=True)
    health_scores, dates = [], []

    start_idx = max(60, int(len(price_df) * 0.05))
    for i in range(start_idx, len(price_df)):
        current_date = price_df.loc[i, "date"]
        cot_slice = cot_df[cot_df["report_date"] <= current_date]
        price_slice = price_df.iloc[: i + 1].copy()
        price_slice = calculate_rvol(price_slice)

        if len(cot_slice) < 2 or len(price_slice) < 30:
            continue

        score = calculate_health_gauge(cot_slice, price_slice)
        health_scores.append(score)
        dates.append(current_date)

    return pd.DataFrame({"date": dates, "health_gauge": health_scores})

# --- Signal generation (entry/exit logic) ---
def generate_signals(health_df: pd.DataFrame, price_df: pd.DataFrame,
                     buy_threshold: float = 7.0, sell_threshold: float = 3.0,
                     rvol_threshold: float = 1.5) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    if health_df is not None and not health_df.empty:
        df = pd.merge(df, health_df, on="date", how="left")
        df["health_gauge"].fillna(method="ffill", inplace=True)
    else:
        df["health_gauge"] = 5.0

    if "rvol" not in df.columns:
        df = calculate_rvol(df)
    if "extension" not in df.columns:
        df = calculate_price_bands(df)

    df["buy_signal"] = (df["health_gauge"] >= buy_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] < 0)
    df["sell_signal"] = (df["health_gauge"] <= sell_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] > 0)
    df["strong_buy"] = df["buy_signal"] & (df["rvol"] > 2.0) & (df["extension"] < -0.5)
    df["strong_sell"] = df["sell_signal"] & (df["rvol"] > 2.0) & (df["extension"] > 0.5)

    return df

# --- Exit Timing Neural Network ---
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

# --- Feature preparation ---
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
    if "health_gauge" in df.columns:
        df["health_change"] = df["health_gauge"].diff()

    df.dropna(inplace=True)
    return df

# --- Train exit model ---
def train_exit_model(df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    df = prepare_features(df.copy())
    if df.empty or len(df) < 50:
        raise ValueError("Not enough data to train exit model")

    df["future_returns_5d"] = df["close"].pct_change(5).shift(-5)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("No data after computing future returns")

    feature_cols = [
        "returns", "log_returns", "rolling_mean_5", "rolling_mean_20",
        "rolling_std_5", "rolling_std_20", "momentum_5", "momentum_20"
    ]
    for c in ["commercial_net", "non_commercial_net", "comm_net_change", "non_comm_net_change", "health_gauge", "extension"]:
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

# ===== end Chunk 2/3 =====

# ===== Chunk 3/3 =====

# --- Performance Metrics ---
def calculate_metrics(df: pd.DataFrame) -> dict:
    returns = df["returns"].fillna(0)
    total_return = df["total"].iloc[-1] / df["total"].iloc[0] - 1 if len(df) > 0 and df["total"].iloc[0] != 0 else 0
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1 if len(df) > 0 else 0
    annualized_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    drawdown = df["total"].cummax() - df["total"] if "total" in df.columns else pd.Series([0])
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

# --- Plot Equity Curve ---
def plot_equity(df: pd.DataFrame, symbol: str):
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["total"], label=f"{symbol} Equity Curve")
    plt.title(f"Equity Curve for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# --- Backtesting with Neural Network ---
def backtest_strategy_with_nn(df: pd.DataFrame, cot_df: pd.DataFrame = pd.DataFrame(),
                              exit_model=None, feature_scaler=None, target_scaler=None,
                              exit_days: int = 5) -> pd.DataFrame:
    df = df.copy()
    df = prepare_features(df)
    df["position"] = 0
    df["exit_countdown"] = 0
    current_position = 0
    exit_timer = 0

    for i in range(1, len(df)):
        # Entry signals
        if df["strong_buy"].iloc[i] and current_position <= 0:
            current_position = 1
            exit_timer = 0
        elif df["strong_sell"].iloc[i] and current_position >= 0:
            current_position = -1
            exit_timer = 0
        elif df["buy_signal"].iloc[i] and current_position <= 0 and not df["strong_sell"].iloc[i]:
            current_position = 1
            exit_timer = 0
        elif df["sell_signal"].iloc[i] and current_position >= 0 and not df["strong_buy"].iloc[i]:
            current_position = -1
            exit_timer = 0

        # Exit timing countdown
        if exit_timer > 0:
            exit_timer -= 1
            if exit_timer == 0:
                current_position = 0

        # Neural network-based exit
        if exit_model is not None and i > 20:
            feature_cols = ['returns', 'log_returns', 'rolling_mean_5', 'rolling_mean_20',
                            'rolling_std_5', 'rolling_std_20', 'momentum_5', 'momentum_20']
            if 'comm_net_change' in df.columns:
                feature_cols.extend(['commercial_net', 'non_commercial_net', 'comm_net_change', 'non_comm_net_change'])
            feature_cols.extend(['position', 'buy_signal', 'sell_signal'])
            df.loc[:, 'position'] = current_position
            try:
                features = df.iloc[i][feature_cols].values.reshape(1, -1)
                features_scaled = feature_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                with torch.no_grad():
                    prediction_scaled = exit_model(features_tensor).numpy()
                    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
                if (current_position > 0 and prediction < -0.005) or (current_position < 0 and prediction > 0.005):
                    if exit_timer == 0:
                        exit_timer = exit_days
            except Exception as e:
                logger.error(f"Error in neural network prediction: {e}")

        # Opposite signal triggers
        if current_position > 0 and df["sell_signal"].iloc[i] and exit_timer == 0:
            exit_timer = exit_days
        elif current_position < 0 and df["buy_signal"].iloc[i] and exit_timer == 0:
            exit_timer = exit_days

        df.loc[df.index[i], "position"] = current_position
        df.loc[df.index[i], "exit_countdown"] = exit_timer

    # Compute returns
    df["returns"] = df["close"].pct_change() * df["position"].shift(1)
    df["total"] = (1 + df["returns"]).fillna(1).cumprod()

    return df

# --- Streamlit Main App ---
def main():
    st.title("Trading Strategy Backtester with Neural Network Exit Timing")

    st.sidebar.header("Configuration")
    start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2024, 1, 1))
    exit_days = st.sidebar.slider("Exit Countdown Days", 1, 30, 5)
    use_nn = st.sidebar.checkbox("Use Neural Network for Exit Timing", True)
    selected_asset = st.sidebar.selectbox("Select Asset", list(assets.keys()))

    if st.sidebar.button("Run Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            symbol = assets[selected_asset]
            price_df = fetch_price_data_yahoo(symbol, start_date.isoformat(), end_date.isoformat())
            if price_df.empty:
                st.error(f"No price data for {symbol}")
                return

            cot_df = fetch_cot_data(selected_asset)
            if cot_df.empty:
                st.warning(f"No COT data for {selected_asset}, using price data only.")

            # Neural network training
            exit_model, feature_scaler, target_scaler = None, None, None
            if use_nn and len(price_df) > 100:
                st.info("Training neural network for exit timing...")
                exit_model, feature_scaler, target_scaler = train_exit_model(price_df.copy(), epochs=30)
                st.success("Neural network trained successfully!")

            # Run backtest
            backtest_results = backtest_strategy_with_nn(
                price_df,
                cot_df if not cot_df.empty else pd.DataFrame(),
                exit_model=exit_model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                exit_days=exit_days
            )

            # Display metrics
            metrics = calculate_metrics(backtest_results)
            st.subheader("Backtest Results")
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Performance Metrics")
                for k, v in metrics.items():
                    st.write(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
            with col2:
                st.write("### Position Summary")
                position_counts = backtest_results["position"].value_counts()
                st.write(f"Long Positions: {position_counts.get(1,0)}")
                st.write(f"Short Positions: {position_counts.get(-1,0)}")
                st.write(f"Neutral Positions: {position_counts.get(0,0)}")

            # Plot equity curve
            st.subheader("Equity Curve")
            plot_equity(backtest_results, symbol)

            # Plot positions
            st.subheader("Positions and Signals")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(backtest_results["date"], backtest_results["close"], label="Price")
            ax.scatter(backtest_results.loc[backtest_results["position"]==1, "date"],
                       backtest_results.loc[backtest_results["position"]==1, "close"],
                       color="green", label="Long", marker="^")
            ax.scatter(backtest_results.loc[backtest_results["position"]==-1, "date"],
                       backtest_results.loc[backtest_results["position"]==-1, "close"],
                       color="red", label="Short", marker="v")
            ax.scatter(backtest_results.loc[(backtest_results["position"].shift(1)!=0) & (backtest_results["position"]==0), "date"],
                       backtest_results.loc[(backtest_results["position"].shift(1)!=0) & (backtest_results["position"]==0), "close"],
                       color="black", label="Exit", marker="x")
            ax.set_title(f"Positions for {symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Exit countdown histogram
            if "exit_countdown" in backtest_results.columns:
                non_zero_countdowns = backtest_results[backtest_results["exit_countdown"]>0]["exit_countdown"]
                if len(non_zero_countdowns) > 0:
                    st.subheader("Exit Countdown Distribution")
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.hist(non_zero_countdowns, bins=range(1, exit_days+2), alpha=0.7)
                    ax.set_xlabel("Days until exit")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Exit Countdown Values")
                    ax.grid(True)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
