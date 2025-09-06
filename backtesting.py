# backtesting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import streamlit as st

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Assets Dictionary ---
assets = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "ZW=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}

# --- Neural Network for Exit Timing ---
class ExitTimingNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ExitTimingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


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

# --- Fetch Price Data (Yahoo Finance API) ---
def fetch_price_data(symbol: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
    params = {
        "period1": int(datetime.datetime.combine(start_date, datetime.datetime.min.time()).timestamp()),
        "period2": int(datetime.datetime.combine(end_date, datetime.datetime.min.time()).timestamp()),
        "interval": "1d",
        "events": "history"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(response.text))
        df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        return pd.DataFrame()


# --- Fetch COT Data (Placeholder Example) ---
def fetch_cot_data(symbol: str) -> pd.DataFrame:
    # Placeholder for actual COT data fetching logic
    # Return empty if unavailable
    return pd.DataFrame()


# --- Neural Network Training ---
def train_exit_model(df: pd.DataFrame, epochs=20):
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"].fillna(0))
    df["rolling_mean_5"] = df["close"].rolling(5).mean()
    df["rolling_mean_20"] = df["close"].rolling(20).mean()
    df["rolling_std_5"] = df["close"].rolling(5).std()
    df["rolling_std_20"] = df["close"].rolling(20).std()
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)

    df.dropna(inplace=True)
    features = df[["returns", "log_returns", "rolling_mean_5", "rolling_mean_20",
                   "rolling_std_5", "rolling_std_20", "momentum_5", "momentum_20"]]
    target = df["returns"].shift(-1).dropna()

    features = features.iloc[:-1]
    target = target.iloc[:len(features)]

    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(features)
    y_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    model = ExitTimingNN(input_size=X_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    return model, feature_scaler, target_scaler


# --- Backtesting Function ---
def backtest_strategy_with_nn(df: pd.DataFrame,
                              cot_df: pd.DataFrame,
                              exit_model=None,
                              feature_scaler=None,
                              target_scaler=None,
                              exit_days=5):
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log1p(df["returns"].fillna(0))
    df["rolling_mean_5"] = df["close"].rolling(5).mean()
    df["rolling_mean_20"] = df["close"].rolling(20).mean()
    df["rolling_std_5"] = df["close"].rolling(5).std()
    df["rolling_std_20"] = df["close"].rolling(20).std()
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_20"] = df["close"].pct_change(20)

    df["buy_signal"] = (df["close"] > df["rolling_mean_5"]) & (df["momentum_5"] > 0)
    df["sell_signal"] = (df["close"] < df["rolling_mean_5"]) & (df["momentum_5"] < 0)
    df["strong_buy"] = (df["buy_signal"]) & (df["momentum_20"] > 0)
    df["strong_sell"] = (df["sell_signal"]) & (df["momentum_20"] < 0)

    df["position"] = 0
    df["exit_countdown"] = 0
    current_position = 0
    exit_timer = 0


# Main trading loop
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

        # Risk management exits
        if current_position > 0 and df["momentum_20"].iloc[i] < 0:
            current_position = 0
        elif current_position < 0 and df["momentum_20"].iloc[i] > 0:
            current_position = 0

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
