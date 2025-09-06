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

# --- Fetch historical price data --- 
def fetch_price_data(ticker_symbol: str, start_date=None, end_date=None, period="2y", interval="1d") -> pd.DataFrame:
    logger.info(f"Fetching price data for {ticker_symbol}")
    tk = Ticker(ticker_symbol)
    
    # Use date range if provided, otherwise use period
    if start_date and end_date:
        df = tk.history(start=start_date, end=end_date, interval=interval).reset_index()
    else:
        df = tk.history(period=period, interval=interval).reset_index()
        
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].sort_values("date")
    return df

# --- Feature Engineering ---
def prepare_features(df):
    # Calculate technical indicators as features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']).diff()
    
    # Add rolling statistics
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
    df['rolling_std_5'] = df['close'].rolling(window=5).std()
    df['rolling_std_20'] = df['close'].rolling(window=20).std()
    
    # Momentum features
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
    
    # COT features (if available)
    if 'commercial_net' in df.columns:
        df['comm_net_change'] = df['commercial_net'].diff()
        df['non_comm_net_change'] = df['non_commercial_net'].diff()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# --- Train Neural Network for Exit Timing ---
def train_exit_model(df, epochs=50):
    # Prepare features
    feature_cols = ['returns', 'log_returns', 'rolling_mean_5', 'rolling_mean_20', 
                    'rolling_std_5', 'rolling_std_20', 'momentum_5', 'momentum_20']
    
    if 'comm_net_change' in df.columns:
        feature_cols.extend(['commercial_net', 'non_commercial_net', 'comm_net_change', 'non_comm_net_change'])
    
    # Add position and signal features
    feature_cols.extend(['position', 'buy_signal', 'sell_signal'])
    
    # Create target - the optimal exit timing (simulated here with future returns)
    # For simplicity, we'll use 5-day future returns as our target
    df['future_returns_5d'] = df['close'].pct_change(5).shift(-5)
    df.dropna(inplace=True)
    
    X = df[feature_cols].values
    y = df['future_returns_5d'].values.reshape(-1, 1)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = ExitTimingModel(input_size=len(feature_cols))
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
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
        
        if (epoch+1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor).item()
        logger.info(f'Test Loss: {test_loss:.4f}')
    
    return model, scaler_X, scaler_y
# --- Backtest Strategy with Neural Network Exit Timing ---
def backtest_strategy_with_nn(price_df, cot_df, exit_model=None, feature_scaler=None, target_scaler=None, exit_days=5):
    df = price_df.copy()
    df = df.merge(cot_df[["report_date", "commercial_net", "non_commercial_net"]],
                  left_on="date", right_on="report_date", how="left")
    df.fillna(method="ffill", inplace=True)
    
    # --- Basic Signals ---
    df["buy_signal"] = df["non_commercial_net"] > 0
    df["sell_signal"] = df["non_commercial_net"] < 0
    
    # --- Initialize positions ---
    df["position"] = 0
    df["exit_countdown"] = 0
    current_position = 0  # 1 for long, -1 for short
    exit_timer = 0
    
    # Prepare features for NN if available
    if exit_model is not None:
        df = prepare_features(df)
    
    for i in range(1, len(df)):
        # Default logic
        if df["buy_signal"].iloc[i] and current_position <= 0:
            # Enter long, exit short
            current_position = 1
            exit_timer = 0  # Reset exit timer on new position
        elif df["sell_signal"].iloc[i] and current_position >= 0:
            # Enter short, exit long
            current_position = -1
            exit_timer = 0  # Reset exit timer on new position
        
        # Check for exit signal
        if exit_timer > 0:
            exit_timer -= 1
            if exit_timer == 0:
                current_position = 0  # Exit position when timer reaches zero
        
        # Neural network exit timing if model is available
        if exit_model is not None and i > 20:  # Need enough data for features
            # Extract features for current row
            feature_cols = ['returns', 'log_returns', 'rolling_mean_5', 'rolling_mean_20', 
                          'rolling_std_5', 'rolling_std_20', 'momentum_5', 'momentum_20']
            
            if 'comm_net_change' in df.columns:
                feature_cols.extend(['commercial_net', 'non_commercial_net', 'comm_net_change', 'non_comm_net_change'])
            
            # Add position and signal
            df.loc[:, 'position'] = current_position  # Update position column
            feature_cols.extend(['position', 'buy_signal', 'sell_signal'])
            
            try:
                # Predict optimal exit timing
                features = df.iloc[i][feature_cols].values.reshape(1, -1)
                features_scaled = feature_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                
                with torch.no_grad():
                    prediction_scaled = exit_model(features_tensor).numpy()
                    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
                
                # If model predicts negative returns, consider starting exit countdown
                if (current_position > 0 and prediction < -0.005) or (current_position < 0 and prediction > 0.005):
                    if exit_timer == 0:  # Only set timer if not already counting down
                        exit_timer = exit_days
            except Exception as e:
                logger.error(f"Error in neural network prediction: {e}")
        
        # Update exit countdown for visualization
        df.loc[df.index[i], "exit_countdown"] = exit_timer
        
        # If opposite signal triggers, consider exit based on neural network
        if current_position > 0 and df["sell_signal"].iloc[i]:
            if exit_timer == 0:  # Only set timer if not already counting down
                exit_timer = exit_days
        elif current_position < 0 and df["buy_signal"].iloc[i]:
            if exit_timer == 0:  # Only set timer if not already counting down
                exit_timer = exit_days
        
        # Set position
        df.loc[df.index[i], "position"] = current_position
    
    # --- Compute Returns ---
    df["returns"] = df["close"].pct_change() * df["position"].shift(1)
    df["total"] = (1 + df["returns"]).fillna(1).cumprod()
    
    return df

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

# --- Main Function ---
def main():
    st.title("Trading Strategy Backtester with Neural Network Exit Timing")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.date(2024, 1, 1))
    exit_days = st.sidebar.slider("Exit Countdown Days", 1, 30, 5)
    use_nn = st.sidebar.checkbox("Use Neural Network for Exit Timing", True)
    selected_asset = st.sidebar.selectbox("Select Asset", list(assets.keys()))
    
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Fetching data and running backtest..."):
            symbol = assets[selected_asset]
            
            # Fetch data
            price_df = fetch_price_data(symbol, start_date=start_date, end_date=end_date)
            if price_df.empty:
                st.error(f"No price data for {symbol}")
                return
                
            cot_df = fetch_cot_data(selected_asset)
            if cot_df.empty:
                st.warning(f"No COT data for {selected_asset}, using price data only.")
                
            # Prepare data
            data = price_df.merge(
                cot_df[["report_date", "commercial_net", "non_commercial_net"]],
                left_on="date",
                right_on="report_date",
                how="left"
            ) if not cot_df.empty else price_df.copy()
            
            data.fillna(method="ffill", inplace=True)
            
            # Generate basic signals
            data["buy_signal"] = data["non_commercial_net"] > 0 if "non_commercial_net" in data.columns else data["close"].pct_change() > 0
            data["sell_signal"] = data["non_commercial_net"] < 0 if "non_commercial_net" in data.columns else data["close"].pct_change() < 0
            
            # Neural network for exit timing
            exit_model = None
            feature_scaler = None
            target_scaler = None
            
            if use_nn and len(data) > 100:  # Need enough data for training
                st.info("Training neural network for exit timing...")
                exit_model, feature_scaler, target_scaler = train_exit_model(data.copy(), epochs=30)
                st.success("Neural network trained successfully!")
                
                # Run backtest with neural network
                backtest_results = backtest_strategy_with_nn(
                    price_df, 
                    cot_df if not cot_df.empty else pd.DataFrame(), 
                    exit_model=exit_model,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    exit_days=exit_days
                )
            else:
                # Run standard backtest
                backtest_results = backtest_strategy_with_nn(
                    price_df, 
                    cot_df if not cot_df.empty else pd.DataFrame(),
                    exit_days=exit_days
                )
            
            # Calculate metrics
            metrics = calculate_metrics(backtest_results)
            
            # Display results
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
            
            # Plot equity curve
            st.subheader("Equity Curve")
            plot_equity(backtest_results, symbol)
            
            # Show positions and signals
            st.subheader("Positions and Signals")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(backtest_results["date"], backtest_results["close"], label="Price")
            ax.scatter(
                backtest_results.loc[backtest_results["position"] == 1, "date"],
                backtest_results.loc[backtest_results["position"] == 1, "close"],
                color="green", label="Long", marker="^"
            )
            ax.scatter(
                backtest_results.loc[backtest_results["position"] == -1, "date"],
                backtest_results.loc[backtest_results["position"] == -1, "close"],
                color="red", label="Short", marker="v"
            )
            ax.scatter(
                backtest_results.loc[(backtest_results["position"].shift(1) != 0) & (backtest_results["position"] == 0), "date"],
                backtest_results.loc[(backtest_results["position"].shift(1) != 0) & (backtest_results["position"] == 0), "close"],
                color="black", label="Exit", marker="x"
            )
            ax.set_title(f"Positions for {symbol}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Show exit countdown histogram
            if "exit_countdown" in backtest_results.columns:
                non_zero_countdowns = backtest_results[backtest_results["exit_countdown"] > 0]["exit_countdown"]
                if len(non_zero_countdowns) > 0:
                    st.subheader("Exit Countdown Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(non_zero_countdowns, bins=range(1, exit_days+2), alpha=0.7)
                    ax.set_xlabel("Days until exit")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Distribution of Exit Countdown Values")
                    ax.grid(True)
                    st.pyplot(fig)

# --- Run Main ---
if __name__ == "__main__":
    main()
    
