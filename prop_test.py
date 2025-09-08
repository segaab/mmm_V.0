import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import plotly.graph_objects as go
from sodapy import Socrata
from yahooquery import Ticker

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Config ---
st.set_page_config(page_title="PropFirm Trading Backtester", page_icon="📊", layout="wide")

# --- Header Display ---
def display_prop_firm_header():
    st.title("📈 Prop Firm Trading Backtester")
    st.subheader("Welcome to the Prop Firm Trading Challenge!")
    st.markdown("""
    Backtest your trading strategies according to various prop firm challenges.
    Select your parameters from the sidebar to get started!
    """)

# --- Asset & Contract Setup ---
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

# --- COT API Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Data Fetching Functions ---
def fetch_cot_data(market_name:str,max_attempts:int=3)->pd.DataFrame:
    where_clause=f'market_and_exchange_names="{market_name}"'
    attempt=0
    while attempt<max_attempts:
        try:
            results=client.get("6dca-aqww",where=where_clause,order="report_date_as_yyyy_mm_dd DESC",limit=1500)
            if not results: return pd.DataFrame()
            df=pd.DataFrame.from_records(results)
            df["report_date"] = pd.to_datetime(df.get("report_date_as_yyyy_mm_dd"), errors="coerce")
            df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all"), errors="coerce").fillna(0) - pd.to_numeric(df.get("commercial_short_all"), errors="coerce").fillna(0)
            df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all"), errors="coerce").fillna(0) - pd.to_numeric(df.get("non_commercial_short_all"), errors="coerce").fillna(0)
            df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all"), errors="coerce").fillna(0)
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"COT fetch error {market_name}: {e}")
            attempt+=1
            time.sleep(1)
    return pd.DataFrame()

def fetch_price_data(ticker:str,years:int=3)->pd.DataFrame:
    try:
        ticker_obj=Ticker(ticker)
        start_date=(datetime.datetime.now()-datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
        df=ticker_obj.history(start=start_date)
        if df.empty: return pd.DataFrame()
        if isinstance(df.index,pd.MultiIndex): df=df.reset_index(level=0,drop=True)
        df=df.reset_index()
        df.rename(columns={"date":"Date"}, inplace=True)
        df["Date"]=pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        logger.error(f"Price fetch error {ticker}: {e}")
        return pd.DataFrame()

def merge_cot_price(cot_df:pd.DataFrame, price_df:pd.DataFrame)->pd.DataFrame:
    if cot_df.empty or price_df.empty: return pd.DataFrame()
    merged = pd.merge(price_df, cot_df, left_on="Date", right_on="report_date", how="left").ffill()
    return merged

def calculate_rvol(df:pd.DataFrame, window:int=20)->pd.Series:
    if "volume" not in df.columns: return pd.Series(index=df.index, dtype=float)
    return df["volume"] / df["volume"].rolling(window=window, min_periods=1).mean()

# --- Health Gauge Calculation ---
def calculate_health_gauge(df:pd.DataFrame)->float:
    if df.empty: return float("nan")
    # Open Interest Score
    oi_series = df.get("open_interest_all", pd.Series()).dropna()
    oi_score = (oi_series.iloc[-1]-oi_series.min())/(oi_series.max()-oi_series.min() if oi_series.max()!=oi_series.min() else 1) if len(oi_series)>1 else 0.5
    # COT Score
    df["commercial_net"] = pd.to_numeric(df.get("commercial_net"), errors="coerce")
    df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_net"), errors="coerce")
    st_score = (df["commercial_net"].iloc[-1]-df["commercial_net"].min())/(df["commercial_net"].max()-df["commercial_net"].min() if df["commercial_net"].max()!=df["commercial_net"].min() else 1)
    lt_score = (df["non_commercial_net"].iloc[-1]-df["non_commercial_net"].min())/(df["non_commercial_net"].max()-df["non_commercial_net"].min() if df["non_commercial_net"].max()!=df["non_commercial_net"].min() else 1)
    cot_score = 0.4*st_score + 0.6*lt_score
    # Price & RVol Score
    pv_score = 0.5
    if "close" in df.columns and "volume" in df.columns:
        df["rvol"] = calculate_rvol(df)
        last_ret = df["close"].pct_change().iloc[-1]
        if last_ret >= 0.02: bucket=5
        elif last_ret >= 0.01: bucket=4
        elif last_ret >= -0.01: bucket=3
        elif last_ret >= -0.02: bucket=2
        else: bucket=1
        pv_score = (bucket-1)/4
    return round((0.25*oi_score + 0.35*cot_score + 0.4*pv_score)*10,2)

# --- Signal & Position ---
def generate_signal(health_gauge: float, buy_threshold: float = 6, sell_threshold: float = 4) -> str:
    if np.isnan(health_gauge): return "HOLD"
    if health_gauge >= buy_threshold: return "BUY"
    elif health_gauge <= sell_threshold: return "SELL"
    return "HOLD"

def calculate_lot_size(balance: float, price: float, asset_class: str, symbol: str, risk_per_trade: float=0.01) -> float:
    contract_size = get_contract_size(asset_class, symbol)
    risk_amount = balance * risk_per_trade
    pip_value = contract_size * 0.0001
    if pip_value==0: return 0
    return round(risk_amount / (pip_value * price),2)

def process_assets(selected_assets:list, balance:float, leverage:float, buy_threshold:float, sell_threshold:float) -> dict:
    results={}
    for asset in selected_assets:
        cot_df=fetch_cot_data(asset)
        ticker=assets[asset]
        price_df=fetch_price_data(ticker)
        merged_df=merge_cot_price(cot_df, price_df)
        health_gauge = calculate_health_gauge(merged_df)
        signal = generate_signal(health_gauge, buy_threshold, sell_threshold)
        asset_class, symbol = asset_classes.get(asset, ("FX","default"))
        current_price = price_df["close"].iloc[-1] if not price_df.empty else 1
        lot_size = calculate_lot_size(balance, current_price, asset_class, symbol)
        results[asset] = {"health_gauge":health_gauge, "signal":signal, "lot_size":lot_size, "current_price":current_price}
    return results

# --- Streamlit UI ---
def main():
    display_prop_firm_header()
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        selected_assets = st.multiselect("Select Assets", list(assets.keys()), default=list(assets.keys())[:3])
        years_back = st.slider("Years of Historical Data",1,10,3)
        leverage = st.number_input("Leverage",1.0,100.0,10.0)
        starting_balance = st.number_input("Account Balance",1000.0,1e6,10000.0)
        buy_threshold = st.slider("Buy Threshold (1-10)",1,10,6)
        sell_threshold = st.slider("Sell Threshold (1-10)",1,10,4)

    if st.button("🚀 Run Backtest"):
        with st.spinner("Fetching and processing data..."):
            results = process_assets(selected_assets, starting_balance, leverage, buy_threshold, sell_threshold)

        st.subheader("📊 Trading Signals & Positions")
        results_df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index":"Asset"})
        st.dataframe(results_df)

        st.subheader("🎯 Health Gauge Analysis")
        fig = go.Figure()
        for asset in selected_assets:
            hg = results[asset]["health_gauge"]
            sig = results[asset]["signal"]
            lots = results[asset]["lot_size"]
            if not np.isnan(hg):
                fig.add_trace(go.Bar(
                    x=[asset],
                    y=[hg],
                    name=asset,
                    text=f"Signal: {sig}\nLots: {lots}",
                    textposition="auto",
                    marker_color="green" if sig=="BUY" else "red" if sig=="SELL" else "gray"
                ))
        fig.update_layout(yaxis_title="Health Gauge Score", title="Health Gauge & Trading Signals", height=500)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("💰 Account Summary")
        total_value = starting_balance
        for asset in selected_assets:
            total_value += results[asset]["current_price"] * results[asset]["lot_size"]
        st.metric("Estimated Portfolio Value", f"${total_value:,.2f}")

if __name__=="__main__":
    main()