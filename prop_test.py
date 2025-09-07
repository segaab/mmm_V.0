import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sodapy import Socrata
from yahooquery import Ticker
from datetime import timedelta

# --- Logging ---
logging.basicConfig(level=logg


ing.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page ---
st.set_page_config(page_title="PropFirm Trading Backtester", page_icon="📊", layout="wide")

# --- Function to display header ---
def display_prop_firm_header():
    st.title("📈 Prop Firm Trading Backtester")
    st.subheader("Welcome to the Prop Firm Trading Challenge!")
    st.markdown("""
    This application allows you to backtest your trading strategies according to various prop firm challenges.
    Please select your parameters from the sidebar to get started!
    """)

# --- Custom CSS for styling ---
st.markdown("""
<style>  
.metric-card {  
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  
    padding: 1rem;  
    border-radius: 10px;  
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  
    color: white;  
    text-align: center;  
    margin: 0.5rem 0;  
}  
.profit-positive { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }  
.profit-negative { background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%); }  
.drawdown-warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }  
.challenge-phase {   
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);  
    border-left: 5px solid #00f2fe;  
}  
.stMetric > div > div > div > div {  
    color: #1f2937;  
    font-weight: bold;  
}  
.phase-indicator {  
    padding: 0.5rem 1rem;  
    border-radius: 25px;  
    font-weight: bold;  
    color: white;  
    text-align: center;  
    margin: 0.25rem;  
}  
.phase-1 { background: #ff6b6b; }  
.phase-2 { background: #4ecdc4; }  
.phase-funded { background: #45b7d1; }  
</style>  
""", unsafe_allow_html=True)

# --- Risk Profiles ---
RISK_PROFILES = {
    "QT Prime 2-Step": {
        "name": "QT Prime 2-Step",
        "daily_drawdown_limit": 0.04,
        "max_drawdown_limit": 0.10,
        "phase_1_target": 0.08,
        "phase_2_target": 0.05,
        "phase_3_target": None,
        "max_risk_per_trade": 0.025,
        "min_trading_days": 4,
        "leverage_limits": {
            "FX": 50,
            "INDICES": 20,
            "OIL": 20,
            "METALS": 15,
            "CRYPTO": 1
        },
        "news_trading": False,
        "stop_loss_required": True,
        "layering_allowed": False
    },
    "QT Prime 3-Step": {
        "name": "QT Prime 3-Step",
        "daily_drawdown_limit": 0.04,
        "max_drawdown_limit": 0.10,
        "phase_1_target": 0.06,
        "phase_2_target": 0.06,
        "phase_3_target": 0.06,
        "max_risk_per_trade": 0.025,
        "min_trading_days": 4,
        "leverage_limits": {
            "FX": 50,
            "INDICES": 20,
            "OIL": 20,
            "METALS": 15,
            "CRYPTO": 1
        },
        "news_trading": False,
        "stop_loss_required": True,
        "layering_allowed": False
    },
    "TopOneTrader Pro": {
        "name": "TopOneTrader Pro",
        "daily_drawdown_limit": 0.05,
        "max_drawdown_limit": 0.10,
        "phase_1_target": 0.08,
        "phase_2_target": 0.05,
        "phase_3_target": None,
        "max_risk_per_trade": 0.20,
        "min_trading_days": 1,
        "consistency_rule": 0.50,
        "leverage_limits": {
            "FX": 30,
            "INDICES": 10,
            "OIL": 10,
            "METALS": 10,
            "CRYPTO": 2
        },
        "news_trading": True,
        "stop_loss_required": False,
        "layering_allowed": True,
        "max_lot_size": 20
    }
}

# --- COT API Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets mapping ---
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


# --- Contract sizes per asset class ---
CONTRACT_SIZES = {
    "FX": 100000,
    "OIL": 1000,
    "METALS": 100,
    "INDICES": 50,
    "CRYPTO": 1
}

# --- Function: fetch COT data for given asset ---
def fetch_cot_data(asset_name: str, client=client, limit=200):
    try:
        where_clause = f"market_and_exchange_name = '{asset_name}'"
        results = client.get("6dca-aqww", where=where_clause, limit=limit)
        if not results:
            logger.warning(f"No COT data found for {asset_name}")
            return pd.DataFrame()
        cot_df = pd.DataFrame.from_records(results)
        cot_df['report_date_as_yyyy_mm_dd'] = pd.to_datetime(cot_df['report_date_as_yyyy_mm_dd'])
        cot_df.sort_values("report_date_as_yyyy_mm_dd", inplace=True)
        cot_df["commercial_net"] = cot_df["commercial_long_positions"].astype(float) - cot_df["commercial_short_positions"].astype(float)
        cot_df["noncommercial_net"] = cot_df["noncommercial_long_positions"].astype(float) - cot_df["noncommercial_short_positions"].astype(float)
        return cot_df
    except Exception as e:
        logger.error(f"Error fetching COT data for {asset_name}: {e}")
        return pd.DataFrame()

# --- Function: fetch historical price & volume data from Yahoo ---
def fetch_price_volume(ticker_symbol: str, period="6mo", interval="1d"):
    try:
        t = Ticker(ticker_symbol)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            logger.warning(f"No price data found for {ticker_symbol}")
            return pd.DataFrame()
        hist.reset_index(inplace=True)
        hist.rename(columns={"adjclose": "close", "volume": "volume"}, inplace=True)
        hist["rvol"] = hist["volume"].rolling(14).mean() / hist["volume"].rolling(14).std()
        return hist[["datetime", "close", "volume", "rvol"]]
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker_symbol}: {e}")
        return pd.DataFrame()

# --- Function: calculate health gauge ---
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return float("nan")
    
    if "commercial_net" not in cot_df.columns or "noncommercial_net" not in cot_df.columns:
        logger.warning("COT columns missing")
        return float("nan")
    
    cot_net_score = cot_df["commercial_net"].iloc[-1] / (cot_df["commercial_net"].abs().max())
    rvol_score = price_df["rvol"].iloc[-1] / (price_df["rvol"].max())
    
    health_gauge = (cot_net_score + rvol_score) / 2
    health_gauge = np.clip(health_gauge, -1, 1)
    
    return health_gauge

# --- Function: map asset to ticker symbol ---
def map_asset_to_ticker(asset_name: str):
    return assets.get(asset_name, None)

# --- Function: get latest two COT reports ---
def two_latest_reports(client, asset_name):
    edt_now = datetime.datetime.utcnow() - timedelta(hours=4)
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)
    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)
    
    latest_str = latest_tuesday.strftime('%Y-%m-%d')
    prev_str = previous_tuesday.strftime('%Y-%m-%d')
    
    latest_result = client.get("6dca-aqww", where=f"market_and_exchange_name='{asset_name}' AND report_date_as_yyyy_mm_dd='{latest_str}'")
    previous_result = client.get("6dca-aqww", where=f"market_and_exchange_name='{asset_name}' AND report_date_as_yyyy_mm_dd='{prev_str}'")
    
    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else None
    previous_df = pd.DataFrame.from_records(previous_result) if previous_result else None
    
    return latest_df, previous_df

# --- Function: determine entry signal based on health gauge ---
def determine_signal(health_gauge: float, buy_threshold: float = 0.6, sell_threshold: float = -0.6):
    if np.isnan(health_gauge):
        return "HOLD"
    if health_gauge >= buy_threshold:
        return "BUY"
    elif health_gauge <= sell_threshold:
        return "SELL"
    else:
        return "HOLD"

# --- Function: calculate allowed lot size based on margin and asset class ---
def calculate_lot_size(account_balance: float, leverage: float, price: float, asset_class: str, risk_pct: float = 0.01):
    contract_size = CONTRACT_SIZES.get(asset_class.upper(), 1)
    max_risk_amount = account_balance * risk_pct
    margin_per_contract = price * contract_size / leverage
    lots = max_risk_amount / margin_per_contract
    return max(1, int(lots))

# --- Function: process positions for all assets ---
def process_assets(asset_list, account_balance, leverage, buy_threshold=0.6, sell_threshold=-0.6):
    signals = {}
    for asset_name in asset_list:
        cot_df = fetch_cot_data(asset_name)
        ticker = map_asset_to_ticker(asset_name)
        price_df = fetch_price_volume(ticker) if ticker else pd.DataFrame()
        health_gauge = calculate_health_gauge(cot_df, price_df)
        signal = determine_signal(health_gauge, buy_threshold, sell_threshold)
        asset_class = asset_classes.get(asset_name, "FX")
        current_price = price_df["close"].iloc[-1] if not price_df.empty else 0
        lot_size = calculate_lot_size(account_balance, leverage, current_price, asset_class)
        signals[asset_name] = {
            "signal": signal,
            "health_gauge": health_gauge,
            "lot_size": lot_size,
            "current_price": current_price
        }
    return signals

# --- Example asset class mapping ---
asset_classes = {
    "EUR/USD": "FX",
    "GBP/USD": "FX",
    "Gold": "METALS",
    "Silver": "METALS",
    "Crude Oil": "OIL",
    "S&P 500": "INDICES",
    "Bitcoin": "CRYPTO"
}

# --- Example asset to ticker mapping ---
assets = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "S&P 500": "^GSPC",
    "Bitcoin": "BTC-USD"
}

# --- Streamlit UI for Prop Firm Backtester ---
def main():
    st.set_page_config(page_title="Prop Firm Backtester", layout="wide")
    st.title("📈 Prop Firm Trading Backtester")
    st.subheader("Welcome to the Prop Firm Trading Challenge!")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        selected_assets = st.multiselect("Select Assets", list(assets.keys()), default=list(assets.keys())[:3])
        years_back = st.slider("Years of Historical Data", min_value=1, max_value=10, value=3)
        leverage = st.number_input("Leverage", min_value=1.0, value=10.0)
        starting_balance = st.number_input("Account Balance", min_value=1000.0, value=10000.0)
        buy_threshold = st.slider("Buy Threshold", min_value=0.0, max_value=1.0, value=0.6)
        sell_threshold = st.slider("Sell Threshold", min_value=-1.0, max_value=0.0, value=-0.6)

    if st.button("🚀 Run Backtest"):
        with st.spinner("Fetching and processing data..."):
            results = process_assets(selected_assets, starting_balance, leverage, buy_threshold, sell_threshold)

        # Display signals and lot sizes
        st.subheader("📊 Trading Signals & Positions")
        results_df = pd.DataFrame.from_dict(results, orient="index")
        results_df.reset_index(inplace=True)
        results_df.rename(columns={"index": "Asset"}, inplace=True)
        st.dataframe(results_df)

        # Plot Health Gauge
        st.subheader("🎯 Health Gauge Analysis")
        fig = go.Figure()
        for asset in selected_assets:
            if results[asset]["health_gauge"] is not None:
                fig.add_trace(go.Bar(
                    x=[asset],
                    y=[results[asset]["health_gauge"]],
                    name=asset,
                    text=f"Signal: {results[asset]['signal']}\nLots: {results[asset]['lot_size']}",
                    textposition="auto"
                ))
        fig.update_layout(
            yaxis_title="Health Gauge Score",
            title="Health Gauge & Trading Signals",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.subheader("💰 Account Summary")
        total_value = starting_balance
        for asset in selected_assets:
            total_value += results[asset]["current_price"] * results[asset]["lot_size"]
        st.metric("Estimated Portfolio Value", f"${total_value:,.2f}")

if __name__ == "__main__":
    main()