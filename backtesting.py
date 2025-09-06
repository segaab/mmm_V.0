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

# --- Helper Functions for Backtesting ---
def load_price_data(asset_name, years_back):
    ticker = assets[asset_name]
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=years_back*365)
    return fetch_yahooquery_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

def load_cot_data(asset_name):
    return fetch_cot_data(asset_name)

def merge_price_cot(price_df, cot_df):
    if price_df.empty or cot_df.empty:
        return pd.DataFrame()
    df = pd.merge(price_df, cot_df[["report_date", "commercial_net", "non_commercial_net"]],
                  left_on="date", right_on="report_date", how="left")
    df.fillna(0, inplace=True)
    return df

def calculate_health_gauge(df):
    if df.empty:
        return df
    df["hg"] = (df["commercial_net"] - df["non_commercial_net"]) / (df["commercial_net"].abs() + df["non_commercial_net"].abs() + 1e-6)
    df["hg"] = df["hg"].clip(-1, 1)
    return df

def generate_signals(df, buy_threshold=0.3, sell_threshold=0.7):
    if df.empty:
        return df
    df["signal"] = 0
    df.loc[df["hg"] > sell_threshold, "signal"] = -1
    df.loc[df["hg"] < buy_threshold, "signal"] = 1
    return df


# --- Execute Backtest ---
def execute_backtest(signals_df: pd.DataFrame, starting_balance=10000, leverage=15,
                     lot_size=1.0, exit_rr=2.0, rr_percent=0.1):
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    balance = starting_balance
    equity_curve = []
    trades = []

    for i in range(1, len(signals_df)):
        signal = signals_df.iloc[i-1]["signal"]
        price_open = signals_df.iloc[i]["close"]
        price_prev = signals_df.iloc[i-1]["close"]

        if signal != 0:
            trade_capital = balance * rr_percent
            trade_return = ((price_open - price_prev) / price_prev) * leverage * signal * lot_size
            balance += trade_capital * trade_return
            rr_actual = trade_return / exit_rr if exit_rr != 0 else 0
            trades.append({
                "date": signals_df.iloc[i]["date"],
                "signal": signal,
                "price": price_open,
                "trade_return": trade_return,
                "rr_actual": rr_actual,
                "balance": balance
            })

        equity_curve.append({"date": signals_df.iloc[i]["date"], "balance": balance})

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    # Full RR metrics
    wins = trades_df[trades_df["trade_return"] > 0].shape[0]
    losses = trades_df[trades_df["trade_return"] <= 0].shape[0]
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    average_rr = trades_df["rr_actual"].mean() if not trades_df.empty else 0
    max_rr = trades_df["rr_actual"].max() if not trades_df.empty else 0
    min_rr = trades_df["rr_actual"].min() if not trades_df.empty else 0

    metrics = {
        "final_balance": balance,
        "total_return": (balance - starting_balance) / starting_balance,
        "num_trades": len(trades_df),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "average_rr": average_rr,
        "max_rr": max_rr,
        "min_rr": min_rr
    }

    return equity_df, trades_df, metrics

# --- Streamlit Interface ---
def main():
    st.title("Health Gauge Trading Strategy Backtester")

    with st.sidebar:
        st.header("Backtest Parameters")

        # Asset selection
        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)

        # Time period
        years_back = st.slider("Years to Backtest", min_value=1, max_value=10, value=3)

        # Signal thresholds
        buy_thresh = st.number_input("Buy Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        sell_thresh = st.number_input("Sell Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        # Position parameters
        lot_size = st.number_input("Lot Size", min_value=0.01, value=1.0, step=0.01)
        exit_rr = st.number_input("Exit RR", min_value=0.1, value=2.0, step=0.1)
        rr_percent = st.number_input("RR % of Capital per Trade", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        starting_balance = st.number_input("Starting Balance", min_value=1000, value=10000, step=1000)
        leverage = st.number_input("Leverage", min_value=1, value=15, step=1)

    # Load data
    price_df = load_price_data(selected_asset, years_back)
    cot_df = load_cot_data(selected_asset)

    merged_df = merge_price_cot(price_df, cot_df)
    hg_df = calculate_health_gauge(merged_df)
    signals_df = generate_signals(hg_df, buy_threshold=buy_thresh, sell_threshold=sell_thresh)

    # Backtest
    equity_df, trades_df, metrics = execute_backtest(signals_df,
                                                     starting_balance=starting_balance,
                                                     leverage=leverage,
                                                     lot_size=lot_size,
                                                     exit_rr=exit_rr,
                                                     rr_percent=rr_percent)

    # Display metrics
    st.subheader("Backtest Metrics")
    st.write(metrics)

    # Plot equity curve
    if not equity_df.empty:
        st.subheader("Equity Curve")
        fig, ax = plt.subplots()
        ax.plot(equity_df["date"], equity_df["balance"], label="Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance")
        ax.legend()
        st.pyplot(fig)

    # Show trades
    if not trades_df.empty:
        st.subheader("Trades Executed")
        st.dataframe(trades_df)

if __name__ == "__main__":
    main()