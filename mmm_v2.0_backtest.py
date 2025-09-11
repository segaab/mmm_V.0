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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets List ---
# --- Assets List (COT names with tickers) ---
assets_list = [
    {"name": "GOLD - COMMODITY EXCHANGE INC.", "symbol": "GC=F"},
    {"name": "SILVER - COMMODITY EXCHANGE INC.", "symbol": "SI=F"},
    {"name": "EURO FX - CHICAGO MERCANTILE EXCHANGE", "symbol": "6E=F"},
    {"name": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "symbol": "6J=F"},
    {"name": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "symbol": "6B=F"},
    {"name": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "symbol": "6C=F"},
    {"name": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "symbol": "6A=F"},
    {"name": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "symbol": "6S=F"},
    {"name": "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "symbol": "ES=F"},
    {"name": "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "symbol": "NQ=F"},
    {"name": "DOW JONES INDUSTRIAL AVERAGE - CHICAGO MERCANTILE EXCHANGE", "symbol": "YM=F"},
    {"name": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE", "symbol": "CL=F"},
    {"name": "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE", "symbol": "NG=F"},
    {"name": "COPPER - COMMODITY EXCHANGE INC.", "symbol": "HG=F"},
    {"name": "PLATINUM - NEW YORK MERCANTILE EXCHANGE", "symbol": "PL=F"},
    {"name": "PALLADIUM - NEW YORK MERCANTILE EXCHANGE", "symbol": "PA=F"},
]

# ------------------------------
# Fetching Functions
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
    logger.info("Fetching Yahoo data for %s", ticker)
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
                hist = hist.reset_index().rename(columns={"index": "date"})
            hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce")
            hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching Yahoo data for %s: %s", ticker, e)
            attempt += 1
            time.sleep(1 + attempt)
    return pd.DataFrame()

# ------------------------------
# Processing Functions
# ------------------------------
def calculate_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["vol_ma"] = df["volume"].rolling(lookback, min_periods=1).mean()
    df["rvol"] = df["volume"] / df["vol_ma"]
    return df

def merge_price_cot(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    if cot_df.empty:
        df["open_interest_all"] = np.nan
        df["commercial_net"] = np.nan
        df["non_commercial_net"] = np.nan
        return df
    cot_latest = cot_df.iloc[-1]
    df["open_interest_all"] = cot_latest.get("open_interest_all", np.nan)
    df["commercial_net"] = cot_latest.get("commercial_net", np.nan)
    df["non_commercial_net"] = cot_latest.get("non_commercial_net", np.nan)
    return df

def calculate_health_gauge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rvol_score"] = np.clip(df["rvol"], 0, 3) / 3
    df["high20"] = df["close"].rolling(20, min_periods=1).max()
    df["low20"] = df["close"].rolling(20, min_periods=1).min()
    df["price_ext"] = (df["close"] - df["low20"]) / (df["high20"] - df["low20"] + 1e-9)
    df["price_score"] = np.clip(df["price_ext"], 0, 1)
    df["cot_score"] = np.tanh(df["commercial_net"] / (df["open_interest_all"] + 1e-9))
    df["health_gauge"] = 0.4 * df["rvol_score"] + 0.3 * df["price_score"] + 0.3 * df["cot_score"]
    df["health_gauge"] = df["health_gauge"].clip(0, 1)
    return df

# ------------------------------
# NN Preparation Functions
# ------------------------------
def prepare_nn_dataset(price_df: pd.DataFrame, health_threshold_buy=0.6, health_threshold_sell=0.4, rr_threshold=0.02):
    df = price_df.copy()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['label'] = 0
    df.loc[(df['health_gauge'] > health_threshold_buy) & (df['future_return'] > rr_threshold), 'label'] = 1
    df.loc[(df['health_gauge'] < health_threshold_sell) & (df['future_return'] < -rr_threshold), 'label'] = 1
    feature_cols = ['rvol', 'price_score', 'cot_score', 'health_gauge', 'close', 'volume']
    df = df.dropna(subset=feature_cols + ['label'])
    X = df[feature_cols].values
    y = df['label'].values
    return X, y

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EntryNN(nn.Module):
    def __init__(self, input_dim):
        super(EntryNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


# ------------------------------
# Chunk 2: NN Entry, Backtester & Streamlit UI
# ------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

# ------------------------------
# Dataset Preparation
# ------------------------------

class TradingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, features: list, target_col: str):
        self.scaler = MinMaxScaler()
        self.features = features
        self.data = data.copy()
        self.X = self.scaler.fit_transform(self.data[features].fillna(0))
        self.y = self.data[target_col].fillna(0).values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ------------------------------
# NN Model
# ------------------------------

class EntryNN(nn.Module):
    def __init__(self, input_dim):
        super(EntryNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# ------------------------------
# Training Function
# ------------------------------

def train_nn(df: pd.DataFrame, features: list, target_col: str, epochs=500, batch_size=32, lr=0.001):
    dataset = TradingDataset(df, features, target_col)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = EntryNN(len(features))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.5f}")
    
    # Validation metrics
    model.eval()
    y_preds, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch).squeeze()
            y_preds.extend(output.numpy())
            y_true.extend(y_batch.numpy())
    
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    win_rate = np.mean((y_preds > 0.5) == (y_true > 0.5))
    rr = np.mean((y_preds - y_true) / (np.abs(y_true) + 1e-9))
    sharpe = np.mean(y_preds - y_true) / (np.std(y_preds - y_true) + 1e-9)
    
    print(f"Validation Win Rate: {win_rate:.3f}, R/R: {rr:.3f}, Sharpe: {sharpe:.3f}")
    return model

# ------------------------------
# Backtesting Function
# ------------------------------

def backtest(df: pd.DataFrame, model: nn.Module, features: list, health_threshold_buy=0.6, health_threshold_sell=0.4):
    df = df.copy()
    df["signal"] = 0
    model.eval()
    with torch.no_grad():
        X = torch.tensor(df[features].fillna(0).values, dtype=torch.float32)
        preds = model(X).squeeze().numpy()
    
    # Apply HealthGauge primer
    df.loc[(preds > 0.5) & (df["health_gauge"] >= health_threshold_buy), "signal"] = 1
    df.loc[(preds <= 0.5) & (df["health_gauge"] <= health_threshold_sell), "signal"] = -1
    
    # Simple returns calculation
    df["returns"] = df["close"].pct_change().shift(-1)  # next day return
    df["strategy_returns"] = df["returns"] * df["signal"]
    df["cum_strategy_returns"] = (1 + df["strategy_returns"]).cumprod()
    df["cum_market_returns"] = (1 + df["returns"]).cumprod()
    
    return df

# ------------------------------
# Streamlit Dashboard
# ------------------------------

st.set_page_config(layout="wide")
st.title("NN + HealthGauge Trading Dashboard")

asset_symbols = [a["symbol"] for a in assets_list]
selected_asset = st.selectbox("Select Asset", asset_symbols)

start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.now())

st.text("Fetching and processing data. This may take a moment...")

cot_data, price_data = fetch_all_data(assets_list, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
df_asset = price_data[selected_asset]

if df_asset is not None and not df_asset.empty:
    st.subheader(f"{selected_asset} Health Gauge")
    fig = px.line(df_asset, x="date", y="health_gauge", title="Health Gauge Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Train NN and backtest
    features = ["rvol", "price_score", "cot_score", "health_gauge"]
    df_asset["target"] = ((df_asset["close"].shift(-1) - df_asset["close"]) > 0).astype(int)
    
    st.text("Training NN for entry signals...")
    model = train_nn(df_asset, features, "target")
    
    st.text("Running backtest...")
    df_bt = backtest(df_asset, model, features)
    
    st.subheader("Strategy vs Market Returns")
    fig2 = px.line(df_bt, x="date", y=["cum_strategy_returns", "cum_market_returns"], labels={"value":"Cumulative Returns", "date":"Date"})
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Performance Metrics")
    total_return = df_bt["cum_strategy_returns"].iloc[-1] - 1
    market_return = df_bt["cum_market_returns"].iloc[-1] - 1
    st.markdown(f"- Strategy Total Return: {total_return:.2%}")
    st.markdown(f"- Market Total Return: {market_return:.2%}")
    st.markdown(f"- Avg Health Gauge: {df_bt['health_gauge'].mean():.2f}")
    
else:
    st.warning("No data available for the selected asset.")