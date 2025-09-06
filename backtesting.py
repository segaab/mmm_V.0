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

# --- Neural Network Model Definition ---
class ExitTimingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(ExitTimingModel, self).__init__()
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
        x = self.layer3(x)
        return x

def calculate_rvol(df, volume_window=20):
    """Calculate Relative Volume (RVOL)"""
    if 'volume' in df.columns:
        df = df.copy()
        df['volume_20ma'] = df['volume'].rolling(volume_window).mean()
        df['rvol'] = df['volume'] / df['volume_20ma']
    return df

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

# --- Fetch historical price data (patched to include volume) ---
def fetch_price_data(ticker_symbol: str, start_date=None, end_date=None, period="2y", interval="1d") -> pd.DataFrame:
    logger.info(f"Fetching price data for {ticker_symbol}")
    tk = Ticker(ticker_symbol)

    # Use date range if provided, otherwise use period
    if start_date and end_date:
        df = tk.history(start=start_date, end=end_date, interval=interval).reset_index()
    else:
        df = tk.history(period=period, interval=interval).reset_index()

    df["date"] = pd.to_datetime(df["date"])
    # Keep close price and volume
    if "close" not in df.columns or "volume" not in df.columns:
        df["close"] = df["adjclose"] if "adjclose" in df.columns else df["close"]
        df["volume"] = df["volume"] if "volume" in df.columns else 0
    df = df[["date", "close", "volume"]].sort_values("date")
    return df

# --- Feature Engineering ---
def prepare_features(df):
    df_features = df.copy()

    # Technical indicators
    df_features['returns'] = df_features['close'].pct_change()
    df_features['log_returns'] = np.log(df_features['close']).diff()
    df_features['rolling_mean_5'] = df_features['close'].rolling(5).mean()
    df_features['rolling_mean_20'] = df_features['close'].rolling(20).mean()
    df_features['rolling_std_5'] = df_features['close'].rolling(5).std()
    df_features['rolling_std_20'] = df_features['close'].rolling(20).std()
    df_features['momentum_5'] = df_features['close'] / df_features['close'].shift(5) - 1
    df_features['momentum_20'] = df_features['close'] / df_features['close'].shift(20) - 1

    # COT features
    if 'commercial_net' in df_features.columns:
        df_features['comm_net_change'] = df_features['commercial_net'].diff()
    if 'non_commercial_net' in df_features.columns:
        df_features['non_comm_net_change'] = df_features['non_commercial_net'].diff()

    # Price extension & health gauge if available
    if 'extension' in df_features.columns:
        df_features['extension_abs'] = df_features['extension'].abs()
    if 'health_gauge' in df_features.columns:
        df_features['health_change'] = df_features['health_gauge'].diff()

    df_features.dropna(inplace=True)
    return df_features

def train_exit_model(df, epochs=50):
    if 'position' not in df.columns:
        df['position'] = 0
    if 'buy_signal' not in df.columns:
        df['buy_signal'] = False
    if 'sell_signal' not in df.columns:
        df['sell_signal'] = False

    feature_cols = ['returns', 'log_returns', 'rolling_mean_5', 'rolling_mean_20',
                    'rolling_std_5', 'rolling_std_20', 'momentum_5', 'momentum_20']

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'returns':
                df['returns'] = df['close'].pct_change()
            elif col == 'log_returns':
                df['log_returns'] = np.log(df['close']).diff()
            elif col == 'rolling_mean_5':
                df['rolling_mean_5'] = df['close'].rolling(5).mean()
            elif col == 'rolling_mean_20':
                df['rolling_mean_20'] = df['close'].rolling(20).mean()
            elif col == 'rolling_std_5':
                df['rolling_std_5'] = df['close'].rolling(5).std()
            elif col == 'rolling_std_20':
                df['rolling_std_20'] = df['close'].rolling(20).std()
            elif col == 'momentum_5':
                df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            elif col == 'momentum_20':
                df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    if 'commercial_net' in df.columns and 'comm_net_change' not in df.columns:
        df['comm_net_change'] = df['commercial_net'].diff()
    if 'non_commercial_net' in df.columns and 'non_comm_net_change' not in df.columns:
        df['non_comm_net_change'] = df['non_commercial_net'].diff()

    if 'commercial_net' in df.columns:
        feature_cols.extend(['commercial_net', 'non_commercial_net', 'comm_net_change', 'non_comm_net_change'])

    feature_cols.extend(['position', 'buy_signal', 'sell_signal'])

    # Target: 5-day future returns
    df['future_returns_5d'] = df['close'].pct_change(5).shift(-5)
    df.dropna(inplace=True)
    if len(df) < 10:
        raise ValueError("Not enough data points after preparing features")

    X = df[feature_cols].values
    y = df['future_returns_5d'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ExitTimingModel(input_size=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Training exit timing model...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor).item()
        logger.info(f'Test Loss: {test_loss:.4f}')

    return model, scaler_X, scaler_y

# --- Backtest Strategy with Neural Network Exit Timing ---
def backtest_strategy_with_nn(price_df, cot_df, exit_model=None, feature_scaler=None, target_scaler=None, exit_days=5):
    if price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()

    # Merge COT data
    if not cot_df.empty:
        df = df.merge(cot_df[["report_date", "commercial_net", "non_commercial_net"]],
                     left_on="date", right_on="report_date", how="left")
        df.fillna(method="ffill", inplace=True)

    # Ensure volume_20ma and rvol exist
    if "volume_20ma" not in df.columns or "rvol" not in df.columns:
        df["volume_20ma"] = df["volume"].rolling(20).mean()
        df["rvol"] = df["volume"] / df["volume_20ma"]

    # Placeholder for extension and health_gauge if missing
    if "extension" not in df.columns:
        df["extension"] = 0.0
    if "health_gauge" not in df.columns:
        df["health_gauge"] = 5.0

    # Generate signals
    buy_threshold, sell_threshold, rvol_threshold = 7.0, 3.0, 1.5
    df["buy_signal"] = (df["health_gauge"] >= buy_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] < 0)
    df["sell_signal"] = (df["health_gauge"] <= sell_threshold) & (df["rvol"] >= rvol_threshold) & (df["extension"] > 0)
    df["strong_buy"] = df["buy_signal"] & (df["rvol"] > 2.0) & (df["extension"] < -0.5)
    df["strong_sell"] = df["sell_signal"] & (df["rvol"] > 2.0) & (df["extension"] > 0.5)

    # Initialize positions
    df["position"] = 0
    df["exit_countdown"] = 0
    current_position, exit_timer = 0, 0

    if exit_model is not None:
        df = prepare_features(df)

    # Trading loop
    for i in range(1, len(df)):
        if df["strong_buy"].iloc[i] and current_position <= 0:
            current_position, exit_timer = 1, 0
        elif df["strong_sell"].iloc[i] and current_position >= 0:
            current_position, exit_timer = -1, 0
        elif df["buy_signal"].iloc[i] and current_position <= 0 and not df["strong_sell"].iloc[i]:
            current_position, exit_timer = 1, 0
        elif df["sell_signal"].iloc[i] and current_position >= 0 and not df["strong_buy"].iloc[i]:
            current_position, exit_timer = -1, 0

        # Countdown exit timer
        if exit_timer > 0:
            exit_timer -= 1
            if exit_timer == 0:
                current_position = 0

        # Neural network exit
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

        # Risk management extreme extension
        if current_position > 0 and df["extension"].iloc[i] > 0.9:
            current_position = 0
        elif current_position < 0 and df["extension"].iloc[i] < -0.9:
            current_position = 0

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

# --- Main ---
def main():
    st.title("Trading Strategy Backtester with Neural Network Exit Timing")

    # Sidebar
    st.sidebar.header("Configuration")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))
    exit_days = st.sidebar.slider("Exit Countdown Days", 1, 30, 5)
    use_nn = st.sidebar.checkbox("Use Neural Network for Exit Timing", True)
    selected_asset = st.sidebar.selectbox("Select Asset", list(assets.keys()))

    if st.sidebar.button("Run Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            symbol = assets[selected_asset]
            price_df = fetch_price_data(symbol, start_date=start_date, end_date=end_date)
            if price_df.empty:
                st.error(f"No price data for {symbol}")
                return

            cot_df = fetch_cot_data(selected_asset)
            if cot_df.empty:
                st.warning(f"No COT data for {selected_asset}, using price data only.")

            # NN exit training
            exit_model, feature_scaler, target_scaler = None, None, None
            if use_nn and len(price_df) > 100:
                st.info("Training neural network for exit timing...")
                exit_model, feature_scaler, target_scaler = train_exit_model(price_df.copy(), epochs=30)
                st.success("Neural network trained successfully!")

            backtest_results = backtest_strategy_with_nn(
                price_df,
                cot_df if not cot_df.empty else pd.DataFrame(),
                exit_model=exit_model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                exit_days=exit_days
            )

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
                st.write(f"Long Positions: {position_counts.get(1, 0)}")
                st.write(f"Short Positions: {position_counts.get(-1, 0)}")
                st.write(f"Neutral Positions: {position_counts.get(0, 0)}")

            st.subheader("Equity Curve")
            plot_equity(backtest_results, symbol)

            st.subheader("Positions and Signals")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(backtest_results["date"], backtest_results["close"], label="Price")
            ax.scatter(backtest_results.loc[backtest_results["position"] == 1, "date"],
                       backtest_results.loc[backtest_results["position"] == 1, "close"],
                       color="green", label="Long", marker="^")
            ax.scatter(backtest_results.loc[backtest_results["position"] == -1, "date"],
                       backtest_results.loc[backtest_results["position"] == -1, "close"],
                       color="red", label="Short", marker="v")
            ax.scatter(backtest_results.loc[(backtest_results["position"].shift(1) != 0) & (backtest_results["position"] == 0), "date"],
                       backtest_results.loc[(backtest_results["position"].shift(1) != 0) & (backtest_results["position"] == 0), "close"],
                       color="black", label="Exit", marker="x")
            ax.set_title(f"Positions for {symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Exit countdown histogram
            if "exit_countdown" in backtest_results.columns:
                non_zero_countdowns = backtest_results[backtest_results["exit_countdown"] > 0]["exit_countdown"]
                if len(non_zero_countdowns) > 0:
                    st.subheader("Exit Countdown Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(non_zero_countdowns, bins=range(1, exit_days + 2), alpha=0.7)
                    ax.set_xlabel("Days until exit")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Exit Countdown Values")
                    ax.grid(True)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()

