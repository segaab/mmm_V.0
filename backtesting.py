# ------------------- CHUNK 1/4 -------------------
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

# ------------------- CHUNK 2/4 -------------------
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

# --- Fetch Price Data ---
def fetch_price_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    logger.info(f"Fetching price data for {ticker}")
    try:
        t = Ticker(ticker)
        df = t.history(start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"date": "date", "close": "close"}, inplace=True)
        return df[['date', 'close']]
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

# --- Batch Fetching for Multithreading ---
def fetch_batch(batch_items, start_date, end_date, cot_results, price_results, lock):
    for asset_name, ticker in batch_items:
        cot_df = fetch_cot_data(asset_name)
        price_df = fetch_price_data(ticker, start_date, end_date)
        with lock:
            cot_results[asset_name] = cot_df
            price_results[asset_name] = price_df


# ------------------- CHUNK 3/4 -------------------
# --- Fetch All Data ---
def fetch_all_data(assets_dict, start_date, end_date, batch_size: int = 5):
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
    signals_list = []

    for asset, df in cot_data.items():
        if df.empty or asset not in price_data:
            continue

        price_df = price_data[asset]
        merged_df = pd.merge(df, price_df, on="date", how="inner")

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
            price = row.get('close', 100)
            signal = row['signal']
            position = balance * pos_fraction * leverage * signal
            pnl = position * (np.random.randn() * 0.01)
            balance += pnl

            equity_curve.append({'date': row['date'], 'asset': asset, 'balance': balance})
            trades.append({'date': row['date'], 'asset': asset, 'signal': signal, 'pnl': pnl})

    return pd.DataFrame(equity_curve), pd.DataFrame(trades), {'final_balance': balance}

# ------------------- CHUNK 4/4 -------------------
# --- Streamlit Backtester UI ---
def main():
    st.title("Health Gauge Trading Strategy Backtester")

    # Sidebar: Backtest Parameters
    with st.sidebar:
        st.header("Backtest Parameters")

        selected_asset = st.selectbox(
            "Select Asset",
            list(assets.keys()),
            index=0
        )

        years_back = st.slider(
            "Years to Backtest",
            min_value=1,
            max_value=10,
            value=5
        )

        start_date = datetime.date.today() - datetime.timedelta(days=365 * years_back)
        end_date = datetime.date.today()

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

        cot_data, price_data = fetch_all_data(
            {selected_asset: assets[selected_asset]},
            start_date=start_date,
            end_date=end_date
        )

        signals_df = process_signals(
            cot_data,
            price_data,
            strategy_params={'net_position_threshold': net_pos_threshold}
        )

        equity_curve_df, trades_df, summary = execute_backtest(
            signals_df,
            starting_balance=10000,
            leverage=leverage,
            position_size=position_size
        )

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