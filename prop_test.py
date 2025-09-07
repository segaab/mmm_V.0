import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import matplotlib.pyplot as plt
from sodapy import Socrata
from yahooquery import Ticker
from datetime import timedelta

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page ---
st.set_page_config(page_title="Trading Strategy Backtester", page_icon="📈", layout="wide")

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

# --- Contract Sizes ---
CONTRACT_SIZES = {
    "FX": {"default": 100000},
    "OIL": {"WTI": 1000, "BRENT": 1000, "default": 1000},
    "METALS": {"XAUUSD": 100, "XAGUSD": 5000, "default": 100},
    "INDICES": {"SP500": 10, "DAX30": 25, "FTSE100": 10, "default": 10}
}

def get_contract_size(asset_class, symbol=None):
    asset_class = asset_class.upper()
    if asset_class not in CONTRACT_SIZES:
        raise ValueError(f"Unknown asset class: {asset_class}")

    if symbol and symbol.upper() in CONTRACT_SIZES[asset_class]:
        return CONTRACT_SIZES[asset_class][symbol.upper()]
    return CONTRACT_SIZES[asset_class]["default"]

# --- Asset Class Mapping ---
asset_classes = {
    "GOLD - COMMODITY EXCHANGE INC.": ("METALS", "XAUUSD"),
    "SILVER - COMMODITY EXCHANGE INC.": ("METALS", "XAGUSD"),
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": ("FX", "EURUSD"),
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": ("FX", "USDJPY"),
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": ("INDICES", "SP500"),
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": ("INDICES", "SP500"),
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE": ("OIL", "WTI"),
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": ("OIL", "default"),
}

# --- Fetch COT data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500
            )
            if not results:
                logger.warning(f"No COT data found for {market_name}")
                return pd.DataFrame()
                
            df = pd.DataFrame.from_records(results)
            
            # Ensure required columns exist or create them
            if "report_date_as_yyyy_mm_dd" in df.columns:
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
            else:
                logger.warning(f"Missing report_date_as_yyyy_mm_dd column in COT data for {market_name}")
                df["report_date"] = pd.NaT
                
            # Create commercial_net column
            if "commercial_long_all" in df.columns and "commercial_short_all" in df.columns:
                df["commercial_long_all"] = pd.to_numeric(df["commercial_long_all"], errors="coerce")
                df["commercial_short_all"] = pd.to_numeric(df["commercial_short_all"], errors="coerce")
                df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]
            else:
                logger.warning(f"Missing commercial long/short columns in COT data for {market_name}")
                df["commercial_net"] = np.nan
                
            # Create non_commercial_net column
            if "non_commercial_long_all" in df.columns and "non_commercial_short_all" in df.columns:
                df["non_commercial_long_all"] = pd.to_numeric(df["non_commercial_long_all"], errors="coerce")
                df["non_commercial_short_all"] = pd.to_numeric(df["non_commercial_short_all"], errors="coerce")
                df["non_commercial_net"] = df["non_commercial_long_all"] - df["non_commercial_short_all"]
            else:
                logger.warning(f"Missing non-commercial long/short columns in COT data for {market_name}")
                df["non_commercial_net"] = np.nan
                
            # Convert open interest
            if "open_interest_all" in df.columns:
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
            else:
                logger.warning(f"Missing open_interest_all column in COT data for {market_name}")
                df["open_interest_all"] = np.nan
                
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching COT for {market_name}: {e}")
            attempt += 1
            time.sleep(1)  # Add a small delay before retrying
            
    logger.error(f"Failed to fetch COT for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()


# --- Fetch price data ---
def fetch_price_data_yahoo(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
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
                except:
                    hist = hist.reset_index(level=0, drop=True)
            hist = hist.reset_index()
            if "date" in hist.columns:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
            else:
                hist.index = pd.to_datetime(hist.index)
                hist = hist.reset_index().rename(columns={"index": "date"})
            hist["close"] = pd.to_numeric(hist.get("close", np.nan), errors="coerce")
            hist["volume"] = pd.to_numeric(hist.get("volume", np.nan), errors="coerce")
            return hist.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.error("Error fetching Yahoo data for %s: %s", ticker, e)
            attempt += 1
    logger.error("Failed fetching Yahoo data for %s after %d attempts.", ticker, max_attempts)
    return pd.DataFrame()

# --- Calculate Relative Volume (RVol) ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if df is None or df.empty or "volume" not in df.columns:
        df["rvol"] = np.nan
        return df
    df = df.copy()
    df["rvol"] = df["volume"] / df["volume"].rolling(window, min_periods=1).mean()
    return df

# --- Merge COT + Price ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        logger.warning("Empty COT or price data, cannot merge")
        return pd.DataFrame()
        
    # Ensure required columns exist
    required_cot_cols = ["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]
    for col in required_cot_cols:
        if col not in cot_df.columns:
            logger.warning(f"Missing required column {col} in COT data")
            cot_df[col] = np.nan
            
    cot_small = cot_df[required_cot_cols].copy()
    cot_small.rename(columns={"report_date": "date"}, inplace=True)
    
    # Ensure date columns are datetime
    price_df["date"] = pd.to_datetime(price_df["date"])
    cot_small["date"] = pd.to_datetime(cot_small["date"])
    
    # Merge with backward filling of COT data (since it's weekly)
    merged = pd.merge_asof(
        price_df.sort_values("date"),
        cot_small.sort_values("date"),
        on="date",
        direction="backward"
    )
    
    # Forward fill COT columns
    for col in ["open_interest_all", "commercial_net", "non_commercial_net"]:
        merged[col] = merged[col].ffill()
        
    return merged


# --- Calculate Health Gauge ---
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return float("nan")

    # Rename 'date' to 'report_date' if 'report_date' is missing
    if "report_date" not in cot_df.columns and "date" in cot_df.columns:
        cot_df = cot_df.rename(columns={"date": "report_date"})

    # Make sure required columns exist
    if "commercial_net" not in cot_df.columns:
        logger.warning("commercial_net column missing in COT data")
        return float("nan")
    if "non_commercial_net" not in cot_df.columns:
        logger.warning("non_commercial_net column missing in COT data")
        return float("nan")
    if "open_interest_all" not in price_df.columns:
        logger.warning("open_interest_all column missing in merged data")
        return float("nan")
    
    # Rest of the function remains exactly as in your script...
    df = price_df.copy()
    df["rvol"] = df.get("rvol", np.nan)
    last_date = df["date"].max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open Interest score (25%)
    oi_series = df["open_interest_all"].dropna()
    oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0

    # COT analytics (35%)
    commercial = cot_df[["report_date", "commercial_net"]].dropna(subset=["commercial_net"])
    non_commercial = cot_df[["report_date", "non_commercial_net"]].dropna(subset=["non_commercial_net"])

    short_term = commercial[commercial["report_date"] >= three_months_ago] if not commercial.empty else pd.DataFrame()
    long_term = non_commercial[non_commercial["report_date"] >= one_year_ago] if not non_commercial.empty else pd.DataFrame()

    st_score = 0.0
    if not short_term.empty and len(short_term) > 1:
        min_val = short_term["commercial_net"].min()
        max_val = short_term["commercial_net"].max()
        if max_val > min_val:
            st_score = float((short_term["commercial_net"].iloc[-1] - min_val) / (max_val - min_val))

    lt_score = 0.0
    if not long_term.empty and len(long_term) > 1:
        min_val = long_term["non_commercial_net"].min()
        max_val = long_term["non_commercial_net"].max()
        if max_val > min_val:
            lt_score = float((long_term["non_commercial_net"].iloc[-1] - min_val) / (max_val - min_val))

    cot_score = 0.4 * st_score + 0.6 * lt_score

    # Price + RVol score (40%)
    recent = df[df["date"] >= three_months_ago]
    if recent.empty or "rvol" not in recent.columns or recent["rvol"].isna().all():
        pv_score = 0.0
    else:
        rvol_75 = recent["rvol"].quantile(0.75)
        recent["vol_avg20"] = recent["volume"].rolling(20, min_periods=1).mean()
        recent["vol_spike"] = recent["volume"] > recent["vol_avg20"]
        filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
        if filt.empty:
            pv_score = 0.0
        else:
            last_ret = float(filt["close"].pct_change().iloc[-1]) if len(filt) > 1 else 0.0
            bucket = 5 if last_ret >= 0.02 else 4 if last_ret >= 0.01 else 3 if last_ret >= -0.01 else 2 if last_ret >= -0.02 else 1
            pv_score = (bucket - 1) / 4.0

    return (0.25 * oi_score + 0.35 * cot_score + 0.4 * pv_score) * 10.0


# --- Generate signals ---
def generate_signals(df, buy_threshold=0.3, sell_threshold=0.7):
    if df is None or df.empty:
        return pd.DataFrame()

    health_gauges = []
    for i in range(len(df)):
        date = df.iloc[i]["date"]
        # Get data up to and including current date
        cot_subset = df[df["date"] <= date].copy()
        price_subset = df[df["date"] <= date].copy()
        
        # Calculate health gauge if we have data
        if not cot_subset.empty and not price_subset.empty and "commercial_net" in df.columns and "non_commercial_net" in df.columns:
            hg = calculate_health_gauge(cot_subset, price_subset)
        else:
            hg = np.nan
        health_gauges.append(hg)

    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df["hg"] = health_gauges
    df["hg"] = df["hg"].fillna(0).clip(0, 10) / 10

    df["signal"] = 0
    df.loc[df["hg"] > sell_threshold, "signal"] = -1
    df.loc[df["hg"] < buy_threshold, "signal"] = 1

    return df


# --- Helper Functions for Backtesting ---
def load_price_data(asset_name, years_back):
    ticker = assets.get(asset_name)
    if ticker is None:
        logger.warning(f"No ticker mapping found for {asset_name}.")
        return pd.DataFrame()
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=years_back * 365)
    return fetch_price_data_yahoo(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

def load_cot_data(asset_name):
    return fetch_cot_data(asset_name)

def process_data_for_backtesting(asset_name, years_back):
    price_df = load_price_data(asset_name, years_back)
    cot_df = load_cot_data(asset_name)

    if price_df.empty or cot_df.empty:
        return pd.DataFrame()

    price_df = calculate_rvol(price_df)
    merged_df = merge_cot_price(cot_df, price_df)
    signals_df = generate_signals(merged_df)
    return signals_df

# --- Position Size Calculation ---
def calculate_position_size(account_balance, risk_percent, leverage, stop_loss_pips, 
                           asset_name, price, margin_alloc_pct=0.5, 
                           maintenance_margin_pct=0.1, min_lot=0.01, lot_step=0.01,
                           max_effective_leverage=None, max_total_exposure_pct=None):
    try:
        asset_class, symbol = asset_classes.get(asset_name, ("FX", "default"))
    except:
        logger.warning(f"Unknown asset: {asset_name}, using FX default")
        asset_class, symbol = "FX", "default"

    contract_size = get_contract_size(asset_class, symbol)
    
    pip_size = 0.01 if asset_class != "FX" else 0.0001
    position_value_per_lot = contract_size * price
    margin_per_lot = position_value_per_lot / leverage

    available_margin = account_balance * margin_alloc_pct * (1 - maintenance_margin_pct)
    if available_margin <= 0:
        return 0.0

    max_lots_margin = available_margin / margin_per_lot
    loss_per_lot = contract_size * pip_size * stop_loss_pips
    max_lots_risk = (account_balance * risk_percent) / loss_per_lot if loss_per_lot > 0 else float("inf")

    raw_lots = min(max_lots_margin, max_lots_risk)

    if max_total_exposure_pct is not None:
        max_value_allowed = account_balance * max_total_exposure_pct
        max_lots_exposure = max_value_allowed / position_value_per_lot
        raw_lots = min(raw_lots, max_lots_exposure)

    if max_effective_leverage is not None and max_effective_leverage > 0:
        max_total_value = account_balance * max_effective_leverage
        max_lots_efflev = max_total_value / position_value_per_lot
        raw_lots = min(raw_lots, max_lots_efflev)

    steps = np.floor(raw_lots / lot_step)
    lots = max(0.0, steps * lot_step)
    if lots < min_lot:
        lots = 0.0

    return lots

# --- Execute Backtest ---
def execute_backtest(signals_df: pd.DataFrame, asset_name: str, starting_balance=10000, leverage=15,
                     lot_size=1.0, exit_rr=2.0, rr_percent=0.1, stop_loss_pips=50,
                     margin_alloc_pct=0.5, maintenance_margin_pct=0.1):
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    balance = starting_balance
    equity_curve = []
    trades = []

    for i in range(1, len(signals_df)):
        signal = signals_df.iloc[i-1].get("signal", 0)
        price_open = signals_df.iloc[i].get("close", np.nan)
        price_prev = signals_df.iloc[i-1].get("close", np.nan)

        if signal != 0 and not np.isnan(price_open) and not np.isnan(price_prev):
            position_lots = calculate_position_size(
                account_balance=balance,
                risk_percent=rr_percent,
                leverage=leverage,
                stop_loss_pips=stop_loss_pips,
                asset_name=asset_name,
                price=price_open,
                margin_alloc_pct=margin_alloc_pct,
                maintenance_margin_pct=maintenance_margin_pct
            ) * lot_size

            trade_return = ((price_open - price_prev) / price_prev) * leverage * signal * position_lots
            balance += balance * rr_percent * trade_return
            rr_actual = trade_return / exit_rr if exit_rr != 0 else 0
            trades.append({
                "date": signals_df.iloc[i]["date"],
                "signal": signal,
                "price": price_open,
                "position_lots": position_lots,
                "trade_return": trade_return,
                "rr_actual": rr_actual,
                "balance": balance
            })

        equity_curve.append({"date": signals_df.iloc[i]["date"], "balance": balance})

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)

    wins = trades_df[trades_df["trade_return"] > 0].shape[0] if not trades_df.empty else 0
    losses = trades_df[trades_df["trade_return"] <= 0].shape[0] if not trades_df.empty else 0
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

        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)
        years_back = st.slider("Years to Backtest", min_value=1, max_value=10, value=3)

        buy_thresh = st.number_input("Buy Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        sell_thresh = st.number_input("Sell Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        lot_size = st.number_input("Lot Size Multiplier", min_value=0.01, value=1.0, step=0.01)
        exit_rr = st.number_input("Exit RR", min_value=0.1, value=2.0, step=0.1)
        rr_percent = st.number_input("RR % of Capital per Trade", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1, value=50, step=1)
        starting_balance = st.number_input("Starting Balance", min_value=1000, value=10000, step=1000)
        leverage = st.number_input("Leverage", min_value=1, value=15, step=1)
        
        margin_alloc_pct = st.slider("Margin Allocation %", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        maintenance_margin_pct = st.slider("Maintenance Margin %", min_value=0.05, max_value=0.5, value=0.1, step=0.05)

    with st.spinner("Loading and processing data..."):
        price_df = load_price_data(selected_asset, years_back)
        cot_df = load_cot_data(selected_asset)

        if price_df.empty or cot_df.empty:
            st.error(f"No data available for {selected_asset}. Please try another asset.")
            return

        price_df = calculate_rvol(price_df)
        signals_df = merge_cot_price(cot_df, price_df)
        signals_df = generate_signals(signals_df, buy_threshold=buy_thresh, sell_threshold=sell_thresh)

    equity_df, trades_df, metrics = execute_backtest(
        signals_df,
        asset_name=selected_asset,
        starting_balance=starting_balance,
        leverage=leverage,
        lot_size=lot_size,
        exit_rr=exit_rr,
        rr_percent=rr_percent,
        stop_loss_pips=stop_loss_pips,
        margin_alloc_pct=margin_alloc_pct,
        maintenance_margin_pct=maintenance_margin_pct
    )

    st.subheader("Backtest Metrics")
    st.write(metrics)

    if not equity_df.empty:
        st.subheader("Equity Curve")
        fig, ax = plt.subplots()
        ax.plot(equity_df["date"], equity_df["balance"], label="Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance")
        ax.legend()
        st.pyplot(fig)

    if not signals_df.empty and "hg" in signals_df.columns:
        st.subheader("Health Gauge Over Time")
        fig, ax = plt.subplots()
        ax.plot(signals_df["date"], signals_df["hg"], label="Health Gauge")
        ax.axhline(y=buy_thresh, color='g', linestyle='--', label=f"Buy Threshold ({buy_thresh})")
        ax.axhline(y=sell_thresh, color='r', linestyle='--', label=f"Sell Threshold ({sell_thresh})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Health Gauge Value")
        ax.legend()
        st.pyplot(fig)

    if not trades_df.empty:
        st.subheader("Trades Executed")
        st.dataframe(trades_df)

if __name__ == "__main__":
    main()