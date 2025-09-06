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
def fetch_price_data(ticker_symbol: str, period="2y", interval="1d") -> pd.DataFrame:
    logger.info(f"Fetching price data for {ticker_symbol}")
    tk = Ticker(ticker_symbol)
    df = tk.history(period=period, interval=interval).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "close"]].sort_values("date")
    return df

# --- Backtest Strategy ---
def backtest_strategy(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df = df.merge(cot_df[["report_date", "commercial_net", "non_commercial_net"]],
                  left_on="date", right_on="report_date", how="left")
    df.fillna(method="ffill", inplace=True)
    
    # --- Signals ---
    df["buy_signal"] = df["non_commercial_net"] > 0
    df["sell_signal"] = df["non_commercial_net"] < 0
    
    # --- Initialize positions ---
    df["position"] = 0
    current_position = 0  # 1 for long, -1 for short
    
    for i in range(1, len(df)):
        if df["buy_signal"].iloc[i]:
            # Enter long, exit short if any
            current_position = 1
        elif df["sell_signal"].iloc[i]:
            # Enter short, exit long if any
            current_position = -1
        # If no signal, maintain previous position
        df["position"].iloc[i] = current_position
    
    # --- Compute Returns ---
    df["returns"] = df["close"].pct_change() * df["position"].shift(1)
    df["cumulative_returns"] = (1 + df["returns"]).cumprod()
    
    return df

# --- Plot Backtest Results ---
def plot_backtest(df: pd.DataFrame, asset_name: str):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="date", y="cumulative_returns", data=df, label="Cumulative Returns")
    plt.title(f"Backtest Cumulative Returns: {asset_name}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    st.pyplot(plt)

# --- Performance Metrics (Preserved from initial upload) ---
def calculate_metrics(df: pd.DataFrame) -> dict:
    returns = df["total"].pct_change().fillna(0)
    total_return = df["total"].iloc[-1] / df["total"].iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    drawdown = df["total"].cummax() - df["total"]
    max_drawdown = drawdown.max()
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
    plt.show()

# --- Main Function ---
def main():
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    # Use assets object list from Chunk 1
    symbols = list(assets.values())

    for symbol in symbols:
        price_df = fetch_price_data(symbol, start_date=start_date, end_date=end_date)
        cot_df = fetch_cot_data(symbol)
        if cot_df.empty:
            logger.warning(f"No COT data for {symbol}, skipping.")
            continue

        # Prepare data
        data = price_df.merge(
            cot_df[["report_date", "commercial_net", "non_commercial_net"]],
            left_on="date",
            right_on="report_date",
            how="left"
        )
        data.fillna(method="ffill", inplace=True)

        # Generate signals
        data["buy_signal"] = data["non_commercial_net"] > 0
        data["sell_signal"] = data["non_commercial_net"] < 0

        # --- Backtest Logic with Buy/Sell as Exits ---
        data["position"] = 0
        current_position = 0  # 1 for long, -1 for short

        for i in range(1, len(data)):
            if data["buy_signal"].iloc[i]:
                current_position = 1  # Enter long, exit short
            elif data["sell_signal"].iloc[i]:
                current_position = -1  # Enter short, exit long
            # If no signal, maintain previous position
            data["position"].iloc[i] = current_position

        # Compute returns
        data["returns"] = data["close"].pct_change() * data["position"].shift(1)
        data["total"] = (1 + data["returns"]).cumprod()

        # Calculate metrics
        metrics = calculate_metrics(data)
        print(f"Metrics for {symbol}:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")

        # Plot equity
        plot_equity(data, symbol)

# --- Run Main ---
if __name__ == "__main__":
    main()
