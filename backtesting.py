import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from sodapy import Socrata
from yahooquery import Ticker
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Trading Strategy Backtester",
    page_icon="📈",
    layout="wide"
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Fetch COT Data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
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
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
                try:
                    df["commercial_net"] = df["commercial_long_all"].astype(float) - df["commercial_short_all"].astype(float)
                    df["non_commercial_net"] = df["non_commercial_long_all"].astype(float) - df["non_commercial_short_all"].astype(float)
                except KeyError:
                    df["commercial_net"] = 0.0
                    df["non_commercial_net"] = 0.0
                return df.sort_values("report_date")
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Fetch Yahoo Price Data ---
def fetch_yahooquery_data(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["date"])
            return hist.sort_values("date")
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()
# --- Data Preprocessing ---
def preprocess_data(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or cot_df.empty:
        return pd.DataFrame()
    
    df = price_df.copy()
    df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["date"])
    
    # Merge COT data
    cot_df_subset = cot_df[["report_date", "commercial_net", "non_commercial_net"]].rename(
        columns={"report_date": "Date"}
    )
    df = pd.merge_asof(df.sort_values("Date"), cot_df_subset.sort_values("Date"), on="Date")
    
    # Feature engineering
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"])
    df["volatility"] = df["log_returns"].rolling(window=14).std()
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()
    df["rsi"] = compute_rsi(df["Close"], 14)
    
    df = df.dropna()
    return df

# --- RSI computation ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Scaling features ---
def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled

# --- Generate Buy/Sell Signals ---
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Buy"] = ((df["sma_20"] > df["sma_50"]) & (df["rsi"] < 30) & (df["commercial_net"] > 0)).astype(int)
    df["Sell"] = ((df["sma_20"] < df["sma_50"]) & (df["rsi"] > 70) & (df["commercial_net"] < 0)).astype(int)
    return df

# --- Prepare Torch Dataset ---
def prepare_torch_dataset(df: pd.DataFrame, feature_cols: list, target_col: str):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    return dataset

# --- Simple Neural Network for Signal Optimization ---
class SignalNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SignalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Train Neural Network ---
def train_nn(model, dataset, epochs=50, batch_size=32, lr=0.001):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    return model

# --- Data Preprocessing ---
def preprocess_data(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or cot_df.empty:
        return pd.DataFrame()
    
    df = price_df.copy()
    df = df.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["date"])
    
    # Merge COT data
    cot_df_subset = cot_df[["report_date", "commercial_net", "non_commercial_net"]].rename(
        columns={"report_date": "Date"}
    )
    df = pd.merge_asof(df.sort_values("Date"), cot_df_subset.sort_values("Date"), on="Date")
    
    # Feature engineering
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"])
    df["volatility"] = df["log_returns"].rolling(window=14).std()
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df["sma_50"] = df["Close"].rolling(window=50).mean()
    df["rsi"] = compute_rsi(df["Close"], 14)
    
    df = df.dropna()
    return df

# --- RSI computation ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Scaling features ---
def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled

# --- Generate Buy/Sell Signals ---
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Buy"] = ((df["sma_20"] > df["sma_50"]) & (df["rsi"] < 30) & (df["commercial_net"] > 0)).astype(int)
    df["Sell"] = ((df["sma_20"] < df["sma_50"]) & (df["rsi"] > 70) & (df["commercial_net"] < 0)).astype(int)
    return df

# --- Prepare Torch Dataset ---
def prepare_torch_dataset(df: pd.DataFrame, feature_cols: list, target_col: str):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    return dataset

# --- Simple Neural Network for Signal Optimization ---
class SignalNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SignalNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Train Neural Network ---
def train_nn(model, dataset, epochs=50, batch_size=32, lr=0.001):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    return model

# --- Execute Backtest ---
def execute_backtest(signals_df: pd.DataFrame, starting_balance=10000, leverage=15, position_size='medium'):
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    balance = starting_balance
    position = 0
    trade_log = []
    equity_curve = []

    size_multiplier = {'small': 0.1, 'medium': 0.25, 'large': 0.5}
    position_fraction = size_multiplier.get(position_size, 0.25)

    for idx, row in signals_df.iterrows():
        price = row['Close']
        buy_signal = row['Buy']
        sell_signal = row['Sell']

        # Buy
        if buy_signal and position == 0:
            position = (balance * position_fraction * leverage) / price
            entry_price = price
            balance -= 0  # No immediate cash deduction
            trade_log.append({'Date': row['Date'], 'Type': 'Buy', 'Price': price, 'Position': position})

        # Sell
        elif sell_signal and position > 0:
            profit = (price - entry_price) * position
            balance += profit
            trade_log.append({'Date': row['Date'], 'Type': 'Sell', 'Price': price, 'Position': position, 'Profit': profit})
            position = 0

        # Update equity
        equity = balance + (position * price if position > 0 else 0)
        equity_curve.append({'Date': row['Date'], 'Equity': equity})

    trade_log_df = pd.DataFrame(trade_log)
    equity_curve_df = pd.DataFrame(equity_curve)
    performance = {
        'Starting Balance': starting_balance,
        'Ending Balance': equity_curve_df['Equity'].iloc[-1],
        'Total Profit': equity_curve_df['Equity'].iloc[-1] - starting_balance,
        'Trades Executed': len(trade_log_df)
    }
    return trade_log_df, equity_curve_df, performance

# --- Streamlit UI ---
def main():
    st.title("Health Gauge Trading Strategy Backtester")

    with st.sidebar:
        st.header("Backtest Parameters")

        # Asset selection
        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)

        # Time period
        years_back = st.slider("Years to Backtest", min_value=1, max_value=10, value=3)

        # Neural network parameters
        epochs = st.number_input("NN Training Epochs", min_value=10, max_value=500, value=50)
        batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=32)
        lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")

        position_size = st.selectbox("Position Size", ['small', 'medium', 'large'])

    # Load price & COT data
    price_df = load_price_data(selected_asset, years_back)
    cot_df = load_cot_data(selected_asset)

    if price_df.empty or cot_df.empty:
        st.warning("Data not available for selected asset.")
        return

    # Preprocess data
    df = preprocess_data(price_df, cot_df)
    feature_cols = ["sma_20", "sma_50", "rsi", "volatility", "commercial_net", "non_commercial_net"]
    df_scaled = scale_features(df, feature_cols)

    # Generate signals
    signals_df = generate_signals(df_scaled)

    # Prepare dataset for NN
    dataset = prepare_torch_dataset(signals_df, feature_cols, 'Buy')
    model = SignalNN(len(feature_cols))
    trained_model = train_nn(model, dataset, epochs=epochs, batch_size=batch_size, lr=lr)

    # Execute backtest
    trade_log_df, equity_curve_df, performance = execute_backtest(signals_df, starting_balance=10000,
                                                                  leverage=15, position_size=position_size)

    # Display results
    st.subheader("Performance Summary")
    st.json(performance)

    st.subheader("Equity Curve")
    st.line_chart(equity_curve_df.set_index('Date')['Equity'])

    st.subheader("Trade Log")
    st.dataframe(trade_log_df)

# --- Run Streamlit App ---
if __name__ == "__main__":
    main()