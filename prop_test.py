import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import pytz
import plotly.graph_objects as go
from sodapy import Socrata
from yahooquery import Ticker

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Config & CSS ---
st.set_page_config(page_title="PropFirm Trading Backtester", page_icon="📊", layout="wide")
st.markdown("""
<style>
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color:white; text-align:center; margin:0.5rem 0; }
.profit-positive { background: linear-gradient(135deg,#11998e 0%,#38ef7d 100%); }
.profit-negative { background: linear-gradient(135deg,#fc466b 0%,#3f5efb 100%); }
.drawdown-warning { background: linear-gradient(135deg,#f093fb 0%,#f5576c 100%); }
.challenge-phase { background: linear-gradient(135deg,#4facfe 0%,#00f2fe 100%); border-left:5px solid #00f2fe; }
.stMetric > div > div > div > div { color:#1f2937; font-weight:bold; }
.phase-indicator { padding:0.5rem 1rem; border-radius:25px; font-weight:bold; color:white; text-align:center; margin:0.25rem; }
.phase-1 { background:#ff6b6b; }
.phase-2 { background:#4ecdc4; }
.phase-funded { background:#45b7d1; }
</style>
""", unsafe_allow_html=True)

# --- Risk Profiles ---
RISK_PROFILES = {
    "QT Prime 2-Step": {"daily_drawdown_limit": 0.04, "max_drawdown_limit": 0.10, "phase_1_target": 0.08, "phase_2_target": 0.05, "phase_3_target": None, "max_risk_per_trade": 0.025, "min_trading_days": 4,
                        "leverage_limits": {"FX":50,"INDICES":20,"OIL":20,"METALS":15,"CRYPTO":1}, "news_trading":False, "stop_loss_required":True, "layering_allowed":False},
    "QT Prime 3-Step": {"daily_drawdown_limit": 0.04, "max_drawdown_limit": 0.10, "phase_1_target": 0.06, "phase_2_target":0.06, "phase_3_target":0.06, "max_risk_per_trade": 0.025, "min_trading_days":4,
                        "leverage_limits":{"FX":50,"INDICES":20,"OIL":20,"METALS":15,"CRYPTO":1}, "news_trading":False, "stop_loss_required":True, "layering_allowed":False},
    "TopOneTrader Pro": {"daily_drawdown_limit":0.05,"max_drawdown_limit":0.10,"phase_1_target":0.08,"phase_2_target":0.05,"phase_3_target":None,"max_risk_per_trade":0.20,"min_trading_days":1,
                         "consistency_rule":0.50, "leverage_limits":{"FX":30,"INDICES":10,"OIL":10,"METALS":10,"CRYPTO":2}, "news_trading":True,"stop_loss_required":False,"layering_allowed":True,"max_lot_size":20}
}

# --- Function to Display Header ---
def display_prop_firm_header():
    st.title("📈 Prop Firm Trading Backtester")
    st.subheader("Welcome to the Prop Firm Trading Challenge!")
    st.markdown("""
    This application allows you to backtest your trading strategies according to various prop firm challenges.
    Please select your parameters from the sidebar to get started!
    """)

# --- Assets & Contracts ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.":"GC=F",
    "SILVER - COMMODITY EXCHANGE INC.":"SI=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":"6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":"6J=F",
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE":"ES=F",
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE":"NQ=F",
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE":"CL=F",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE":"NG=F"
}

asset_classes = {
    "GOLD - COMMODITY EXCHANGE INC.":("METALS","XAUUSD"),
    "SILVER - COMMODITY EXCHANGE INC.":("METALS","XAGUSD"),
    "EURO FX - CHICAGO MERCANTILE EXCHANGE":("FX","EURUSD"),
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE":("FX","USDJPY"),
    "S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE":("INDICES","SP500"),
    "NASDAQ-100 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE":("INDICES","SP500"),
    "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE":("OIL","WTI"),
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE":("OIL","default")
}

CONTRACT_SIZES = {
    "FX":{"default":100000},
    "OIL":{"WTI":1000,"BRENT":1000,"default":1000},
    "METALS":{"XAUUSD":100,"XAGUSD":5000,"default":100},
    "INDICES":{"SP500":10,"DAX30":25,"FTSE100":10,"default":10}
}

def get_contract_size(asset_class, symbol=None):
    asset_class = asset_class.upper()
    if asset_class not in CONTRACT_SIZES:
        raise ValueError(f"Unknown asset class: {asset_class}")
    if symbol and symbol.upper() in CONTRACT_SIZES[asset_class]:
        return CONTRACT_SIZES[asset_class][symbol.upper()]
    return CONTRACT_SIZES[asset_class]["default"]

# --- Socrata Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Updated Fetching Functions (UTC-aware) ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get("6dca-aqww", where=where_clause, order="report_date_as_yyyy_mm_dd DESC", limit=1500)
            if not results: return pd.DataFrame()
            df = pd.DataFrame.from_records(results)
            df["report_date"] = pd.to_datetime(df.get("report_date_as_yyyy_mm_dd"), errors="coerce", utc=True)
            df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all",0), errors="coerce") - pd.to_numeric(df.get("commercial_short_all",0), errors="coerce")
            df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all",0), errors="coerce") - pd.to_numeric(df.get("non_commercial_short_all",0), errors="coerce")
            df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", np.nan), errors="coerce")
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching COT for {market_name}: {e}")
            attempt += 1
            time.sleep(1)
    return pd.DataFrame()


def fetch_price_data(ticker: str, years: int = 3) -> pd.DataFrame:
    try:
        ticker_obj = Ticker(ticker)
        start_date = (datetime.datetime.now(pytz.UTC) - datetime.timedelta(days=365 * years)).strftime("%Y-%m-%d")
        df = ticker_obj.history(start=start_date)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex): df = df.reset_index(level=0, drop=True)
        df = df.reset_index().rename(columns={"date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if "volume" not in df.columns: df["volume"] = np.nan
        for col in ["close","open","high","low"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()


def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty: return pd.DataFrame()
    cot_df = cot_df.copy()
    price_df = price_df.copy()
    cot_df["report_date"] = pd.to_datetime(cot_df["report_date"], utc=True)
    price_df["date"] = pd.to_datetime(price_df["date"], utc=True)
    merged = pd.merge_asof(price_df.sort_values("date"), cot_df.sort_values("report_date"), left_on="date", right_on="report_date", direction="backward")
    for col in ["open_interest_all","commercial_net","non_commercial_net"]:
        merged[col] = merged[col].ffill() if col in merged.columns else np.nan
    return merged

# --- Relative Volume Calculation ---
def calculate_rvol(price_df: pd.DataFrame, period: int = 20) -> pd.Series:
    if price_df.empty or "volume" not in price_df.columns:
        return pd.Series(dtype=float)
    vol = price_df["volume"].rolling(window=period, min_periods=1).mean()
    rvol = price_df["volume"] / vol
    return rvol.fillna(1.0)

# --- Health Gauge Calculation ---
def calculate_health_gauge(merged_df: pd.DataFrame) -> pd.DataFrame:
    if merged_df.empty:
        return merged_df
    df = merged_df.copy()
    df["rvol"] = calculate_rvol(df)
    df["health_score"] = (df["commercial_net"].fillna(0) / df["open_interest_all"].replace(0,np.nan)).fillna(0)
    df["health_score"] = df["health_score"] * df["rvol"]
    df["signal"] = 0
    df.loc[df["health_score"] > 0.05, "signal"] = 1
    df.loc[df["health_score"] < -0.05, "signal"] = -1
    return df

# --- Base Lot Size Calculation ---
def calculate_base_lot_size(asset_class: str, symbol: str, account_balance: float, max_risk_pct: float, stop_loss_pips: float) -> float:
    contract_size = get_contract_size(asset_class, symbol)
    risk_amount = account_balance * max_risk_pct
    if stop_loss_pips <= 0: stop_loss_pips = 1
    lot_size = risk_amount / (stop_loss_pips * contract_size)
    return max(lot_size, 0.0)

# --- Generate Trade Signals from Health Gauge ---
def generate_trade_signals(df: pd.DataFrame, max_risk_per_trade: float, account_balance: float, stop_loss_pips: float):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df["lot_size"] = df.apply(lambda row: calculate_base_lot_size(
        asset_classes.get(row.get("market_and_exchange_names"), ("FX","default"))[0],
        asset_classes.get(row.get("market_and_exchange_names"), ("FX","default"))[1],
        account_balance, max_risk_per_trade, stop_loss_pips
    ), axis=1)
    return df

# --- Apply Risk Limits to Signals ---
def apply_risk_limits(trade_df: pd.DataFrame, daily_drawdown_limit: float, max_drawdown_limit: float, account_balance: float) -> pd.DataFrame:
    if trade_df.empty: return trade_df
    df = trade_df.copy()
    df["max_daily_risk"] = account_balance * daily_drawdown_limit
    df["max_total_risk"] = account_balance * max_drawdown_limit
    df["lot_size"] = df[["lot_size"]].clip(upper=df["max_total_risk"])
    return df

# --- Backtest Single Asset ---
def backtest_asset(asset_name: str, ticker_symbol: str, account_balance: float, max_risk_per_trade: float, stop_loss_pips: float, risk_limits: dict):
    cot_df = fetch_cot_data(asset_name)
    price_df = fetch_price_data(ticker_symbol)
    merged_df = merge_cot_price(cot_df, price_df)
    if merged_df.empty:
        logger.warning(f"No merged data for {asset_name}, skipping.")
        return pd.DataFrame()
    merged_df = calculate_health_gauge(merged_df)
    merged_df = generate_trade_signals(merged_df, max_risk_per_trade, account_balance, stop_loss_pips)
    merged_df = apply_risk_limits(merged_df, risk_limits.get("daily_drawdown_limit",0.05),
                                  risk_limits.get("max_drawdown_limit",0.10), account_balance)
    return merged_df


# --- Multi-Asset Backtester ---
def run_multi_asset_backtest(selected_assets, starting_balance, risk_profile_name, stop_loss_pips=50):
    risk_profile = RISK_PROFILES.get(risk_profile_name, {"max_risk_per_trade":0.01, "daily_drawdown_limit":0.05, "max_drawdown_limit":0.10})
    account_balance = starting_balance
    all_results = []

    for asset in selected_assets:
        ticker_symbol = assets.get(asset)
        if not ticker_symbol:
            logger.warning(f"Ticker not found for {asset}, skipping.")
            continue
        asset_df = backtest_asset(asset, ticker_symbol, account_balance, risk_profile.get("max_risk_per_trade",0.01),
                                  stop_loss_pips, risk_profile)
        if asset_df.empty:
            continue
        all_results.append(asset_df)
        # update account_balance sequentially
        pnl = (asset_df["lot_size"] * asset_df["signal"] * asset_df.get("close",1)).sum()
        account_balance += pnl

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return combined_df, account_balance

# --- Streamlit Main Function ---
def main():
    display_prop_firm_header()
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        risk_profile_name = st.selectbox("Risk Profile", list(RISK_PROFILES.keys()), index=0)
        selected_assets = st.multiselect("Select Assets", list(assets.keys()), default=list(assets.keys())[:2])
        starting_balance = st.number_input("Account Balance", min_value=1000.0, value=10000.0)
        stop_loss_pips = st.number_input("Assumed Stop-Loss Pips", min_value=1, value=50)
    if st.button("🚀 Run Backtest"):
        with st.spinner("Running multi-asset backtest..."):
            combined_df, final_balance = run_multi_asset_backtest(selected_assets, starting_balance, risk_profile_name, stop_loss_pips)
        st.subheader("📈 Account & Trade Overview")
        if not combined_df.empty:
            st.dataframe(combined_df)
        st.subheader("💰 Final Account Balance")
        st.metric("Final Balance", f"${final_balance:,.2f}")

# --- Entry Point ---
if __name__ == "__main__":
    main()