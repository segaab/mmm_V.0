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

# --- Assets Mapping (Full COT Assets + Futures Tickers, without RBOB Gasoline and Heating Oil) ---
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

# --- Calculate Relative Volume ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        df["rvol"] = np.nan
        return df
    rolling_avg = df[vol_col].rolling(window).mean()
    df["rvol"] = df[vol_col] / rolling_avg
    return df

# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()
    cot_df_small = cot_df[["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]].copy()
    cot_df_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    cot_df_small = cot_df_small.sort_values("date")
    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")
    merged = pd.merge(price_df, cot_df_filled[["date", "open_interest_all", "commercial_net", "non_commercial_net"]], on="date", how="left")
    merged["open_interest_all"] = merged["open_interest_all"].ffill()
    merged["commercial_net"] = merged["commercial_net"].ffill()
    merged["non_commercial_net"] = merged["non_commercial_net"].ffill()
    return merged


# --- Threaded Batch Fetch ---
def fetch_batch(batch_assets, start_date, end_date, cot_results, price_results, lock):
    for cot_name, ticker in batch_assets:
        try:
            cot_df = fetch_cot_data(cot_name)
            if cot_df.empty:
                with lock:
                    cot_results[cot_name] = pd.DataFrame()
                    price_results[cot_name] = pd.DataFrame()
                continue
            cot_start = cot_df["report_date"].min().date()
            cot_end = cot_df["report_date"].max().date()
            adj_start = max(start_date, cot_start)
            adj_end = min(end_date, cot_end + datetime.timedelta(days=7))
            price_df = fetch_yahooquery_data(ticker, adj_start.isoformat(), adj_end.isoformat())
            if price_df.empty:
                with lock:
                    cot_results[cot_name] = cot_df
                    price_results[cot_name] = pd.DataFrame()
                continue
            price_df = calculate_rvol(price_df)
            merged_df = merge_cot_price(cot_df, price_df)
            with lock:
                cot_results[cot_name] = cot_df
                price_results[cot_name] = merged_df
        except Exception as e:
            logger.error(f"Error loading data for {cot_name}: {e}")
            with lock:
                cot_results[cot_name] = pd.DataFrame()
                price_results[cot_name] = pd.DataFrame()


# --- Fetch All Data ---
def fetch_all_data(assets_dict, start_date, end_date, batch_size: int = 5):
    """
    Fetch COT and price data for all assets in batches using multithreading.
    """
    cot_results = {}
    price_results = {}
    lock = threading.Lock()

    items = list(assets_dict.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    for i in range(0, len(batches), 2):  # process two batches in parallel
        active_threads = []
        for j in range(i, min(i + 2, len(batches))):
            t = threading.Thread(
                target=fetch_batch,
                args=(batches[j], start_date, end_date, cot_results, price_results, lock)
            )
            t.start()
            active_threads.append(t)
        for t in active_threads:
            t.join()

    return cot_results, price_results

# --- Process Signals ---
def process_signals(cot_data, price_data, strategy_params):
    """
    Generate trading signals based on COT net positions and optional strategy parameters.
    """
    signals_list = []

    for asset, df in cot_data.items():
        if df.empty or asset not in price_data:
            continue

        price_df = price_data[asset]
        merged_df = pd.merge(df, price_df, on="date", how="inner")

        # Example strategy: long if non-commercial net position > threshold, short if < -threshold
        net_position = merged_df['non_commercial_long'] - merged_df['non_commercial_short']
        threshold = strategy_params.get('net_position_threshold', 10000)

        merged_df['signal'] = 0
        merged_df.loc[net_position > threshold, 'signal'] = 1
        merged_df.loc[net_position < -threshold, 'signal'] = -1

        signals_list.append(merged_df[['date', 'signal']].assign(asset=asset))

    if signals_list:
        return pd.concat(signals_list, ignore_index=True)
    return pd.DataFrame()

# --- Execute Backtest ---
def execute_backtest(signals_df, starting_balance=10000, leverage=15, position_size='medium'):
    """
    Simple backtester using signals, returns equity curve, trade log, and summary metrics.
    """
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    balance = starting_balance
    equity_curve = []
    trades = []

    size_map = {'small': 0.01, 'medium': 0.05, 'large': 0.1}
    pos_fraction = size_map.get(position_size, 0.05)

    grouped = signals_df.groupby('asset')
    for asset, group in grouped:
        group = group.sort_values('date')
        for _, row in group.iterrows():
            price = row.get('close', 100)  # fallback price
            signal = row['signal']
            position = balance * pos_fraction * leverage * signal
            # For demonstration, simulate PnL with random noise (replace with real price changes)
            pnl = position * (np.random.randn() * 0.01)
            balance += pnl

            equity_curve.append({'date': row['date'], 'asset': asset, 'balance': balance})
            trades.append({'date': row['date'], 'asset': asset, 'signal': signal, 'pnl': pnl})

    return pd.DataFrame(equity_curve), pd.DataFrame(trades), {'final_balance': balance}

# --- Streamlit Backtester UI ---
import streamlit as st
import datetime

def main():
    st.title("Health Gauge Trading Strategy Backtester")

    # Sidebar: Backtest Parameters
    with st.sidebar:
        st.header("Backtest Parameters")

        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset",
            list(COTAssets.keys()),  # COTAssets assumed to be defined elsewhere
            index=0
        )

        # Time period selection
        years_back = st.slider(
            "Years to Backtest",
            min_value=1,
            max_value=10,
            value=5
        )

        start_date = datetime.date.today() - datetime.timedelta(days=365 * years_back)
        end_date = datetime.date.today()

        # Strategy parameters
        net_pos_threshold = st.number_input(
            "Non-Commercial Net Position Threshold",
            value=10000,
            step=1000
        )
        leverage = st.select_slider(
            "Leverage",
            options=[1, 5, 10, 15, 20],
            value=15
        )
        position_size = st.selectbox(
            "Position Size",
            options=['small', 'medium', 'large'],
            index=1
        )

        run_button = st.button("Run Backtest")

    if run_button:
        st.info("Fetching data and running backtest, please wait...")

        # Fetch all data
        cot_data, price_data = fetch_all_data(
            {selected_asset: COTAssets[selected_asset]},
            start_date=start_date,
            end_date=end_date
        )

        # Process signals
        signals_df = process_signals(
            cot_data,
            price_data,
            strategy_params={'net_position_threshold': net_pos_threshold}
        )

        # Execute backtest
        equity_curve_df, trades_df, summary = execute_backtest(
            signals_df,
            starting_balance=10000,
            leverage=leverage,
            position_size=position_size
        )

        # Display results
        st.subheader("Equity Curve")
        if not equity_curve_df.empty:
            st.line_chart(equity_curve_df.set_index('date')['balance'])
        else:
            st.write("No data available for equity curve.")

        st.subheader("Trade Log")
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.write("No trades generated.")

        st.subheader("Summary")
        st.json(summary)

if __name__ == "__main__":
    main()
