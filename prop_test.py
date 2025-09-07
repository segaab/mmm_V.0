import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sodapy import Socrata
from yahooquery import Ticker
from datetime import timedelta

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Function to display header ---
def display_prop_firm_header():
    st.title("📈 Prop Firm Trading Backtester")
    st.subheader("Welcome to the Prop Firm Trading Challenge!")
    st.markdown("""
    This application allows you to backtest your trading strategies according to various prop firm challenges.
    Please select your parameters from the sidebar to get started!
    """)

# --- Streamlit Page ---
st.set_page_config(page_title="PropFirm Trading Backtester", page_icon="📊", layout="wide")

# Custom CSS for prop firm styling
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

# --- Risk Profiles for Prop Firms ---
RISK_PROFILES = {
    "QT Prime 2-Step": {"daily_drawdown_limit": 0.04, "max_drawdown_limit": 0.10, "phase_1_target": 0.08, "phase_2_target": 0.05, "phase_3_target": None, "max_risk_per_trade": 0.025, "min_trading_days": 4,
                        "leverage_limits": {"FX":50,"INDICES":20,"OIL":20,"METALS":15,"CRYPTO":1}, "news_trading":False, "stop_loss_required":True, "layering_allowed":False},
    "QT Prime 3-Step": {"daily_drawdown_limit": 0.04, "max_drawdown_limit": 0.10, "phase_1_target": 0.06, "phase_2_target":0.06, "phase_3_target":0.06, "max_risk_per_trade": 0.025, "min_trading_days":4,
                        "leverage_limits":{"FX":50,"INDICES":20,"OIL":20,"METALS":15,"CRYPTO":1}, "news_trading":False, "stop_loss_required":True, "layering_allowed":False},
    "TopOneTrader Pro": {"daily_drawdown_limit":0.05,"max_drawdown_limit":0.10,"phase_1_target":0.08,"phase_2_target":0.05,"phase_3_target":None,"max_risk_per_trade":0.20,"min_trading_days":1,
                         "consistency_rule":0.50, "leverage_limits":{"FX":30,"INDICES":10,"OIL":10,"METALS":10,"CRYPTO":2}, "news_trading":True,"stop_loss_required":False,"layering_allowed":True,"max_lot_size":20}
}

# --- COT API Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Assets Mapping ---
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

# --- Contract Sizes ---
CONTRACT_SIZES = {"FX":{"default":100000},"OIL":{"WTI":1000,"BRENT":1000,"default":1000},"METALS":{"XAUUSD":100,"XAGUSD":5000,"default":100},"INDICES":{"SP500":10,"DAX30":25,"FTSE100":10,"default":10}}

def get_contract_size(asset_class, symbol=None):
    asset_class = asset_class.upper()
    if asset_class not in CONTRACT_SIZES:
        raise ValueError(f"Unknown asset class: {asset_class}")
    if symbol and symbol.upper() in CONTRACT_SIZES[asset_class]:
        return CONTRACT_SIZES[asset_class][symbol.upper()]
    return CONTRACT_SIZES[asset_class]["default"]

# --- Asset Class Mapping ---
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

# --- Data Fetching Functions ---
def fetch_cot_data(market_name:str,max_attempts:int=3)->pd.DataFrame:
    where_clause=f'market_and_exchange_names="{market_name}"'
    attempt=0
    while attempt<max_attempts:
        try:
            results=client.get("6dca-aqww",where=where_clause,order="report_date_as_yyyy_mm_dd DESC",limit=1500)
            if not results:
                logger.warning(f"No COT data found for {market_name}")
                return pd.DataFrame()
            df=pd.DataFrame.from_records(results)
            if "report_date_as_yyyy_mm_dd" in df.columns:
                df["report_date"]=pd.to_datetime(df["report_date_as_yyyy_mm_dd"],errors="coerce")
            else:
                logger.warning(f"Missing report_date_as_yyyy_mm_dd column in COT data for {market_name}")
                df["report_date"]=pd.NaT
            if "commercial_long_all" in df.columns and "commercial_short_all" in df.columns:
                df["commercial_long_all"]=pd.to_numeric(df["commercial_long_all"],errors="coerce")
                df["commercial_short_all"]=pd.to_numeric(df["commercial_short_all"],errors="coerce")
                df["commercial_net"]=df["commercial_long_all"]-df["commercial_short_all"]
            else:
                logger.warning(f"Missing commercial long/short columns in COT data for {market_name}")
                df["commercial_net"]=np.nan
            if "non_commercial_long_all" in df.columns and "non_commercial_short_all" in df.columns:
                df["non_commercial_long_all"]=pd.to_numeric(df["non_commercial_long_all"],errors="coerce")
                df["non_commercial_short_all"]=pd.to_numeric(df["non_commercial_short_all"],errors="coerce")
                df["non_commercial_net"]=df["non_commercial_long_all"]-df["non_commercial_short_all"]
            else:
                logger.warning(f"Missing non-commercial long/short columns in COT data for {market_name}")
                df["non_commercial_net"]=np.nan
            if "open_interest_all" in df.columns:
                df["open_interest_all"]=pd.to_numeric(df["open_interest_all"],errors="coerce")
            else:
                logger.warning(f"Missing open_interest_all column in COT data for {market_name}")
                df["open_interest_all"]=np.nan
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching COT for {market_name}: {e}")
            attempt+=1
            time.sleep(1)
    logger.error(f"Failed to fetch COT for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()


# --- Price & Volume Fetching ---
def fetch_price_data(ticker:str,years:int=3)->pd.DataFrame:
    try:
        ticker_obj=Ticker(ticker)
        start_date=(datetime.datetime.now()-datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
        df=ticker_obj.history(start=start_date)
        if df.empty:
            logger.warning(f"No price data for {ticker}")
            return pd.DataFrame()
        if isinstance(df.index,pd.MultiIndex):
            df=df.reset_index(level=0,drop=True)
        df=df.reset_index()
        df.rename(columns={"date":"Date"},inplace=True)
        df["Date"]=pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

# --- Merge COT & Price Data ---
def merge_cot_price(cot_df:pd.DataFrame,price_df:pd.DataFrame)->pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()
    merged=pd.merge(price_df,cot_df,left_on="Date",right_on="report_date",how="left")
    merged=merged.ffill()
    return merged

# --- Calculate Relative Volume ---
def calculate_rvol(df:pd.DataFrame,window:int=20)->pd.Series:
    if "volume" not in df.columns:
        logger.warning("Volume column missing in price data")
        return pd.Series(index=df.index,dtype=float)
    rolling_avg=df["volume"].rolling(window=window,min_periods=1).mean()
    return df["volume"]/rolling_avg

# --- Health Gauge Calculation ---
def calculate_health_gauge(df:pd.DataFrame)->float:
    if df.empty:
        return float("nan")

    # Open Interest Score (25%)
    if "open_interest_all" in df.columns:
        oi_series=df["open_interest_all"].dropna()
        if len(oi_series)>1:
            oi_score=(oi_series.iloc[-1]-oi_series.min())/(oi_series.max()-oi_series.min() if oi_series.max()!=oi_series.min() else 1)
        else:
            oi_score=0.5
    else:
        oi_score=0.5

    # COT Score (35%)
    cot_score=0.5
    if "commercial_net" in df.columns and "non_commercial_net" in df.columns:
        df["commercial_net"]=pd.to_numeric(df["commercial_net"],errors="coerce")
        df["non_commercial_net"]=pd.to_numeric(df["non_commercial_net"],errors="coerce")
        short_term=df.tail(12*3)
        long_term=df.tail(12*12)
        if not short_term["commercial_net"].isna().all():
            st_score=(short_term["commercial_net"].iloc[-1]-short_term["commercial_net"].min())/(short_term["commercial_net"].max()-short_term["commercial_net"].min() if short_term["commercial_net"].max()!=short_term["commercial_net"].min() else 1)
        else:
            st_score=0.5
        if not long_term["non_commercial_net"].isna().all():
            lt_score=(long_term["non_commercial_net"].iloc[-1]-long_term["non_commercial_net"].min())/(long_term["non_commercial_net"].max()-long_term["non_commercial_net"].min() if long_term["non_commercial_net"].max()!=long_term["non_commercial_net"].min() else 1)
        else:
            lt_score=0.5
        cot_score=0.4*st_score+0.6*lt_score

    # Price + RVol Score (40%)
    pv_score=0.5
    if "close" in df.columns and "volume" in df.columns:
        df["rvol"]=calculate_rvol(df)
        recent=df.tail(60)
        recent["price_change"]=recent["close"].pct_change()
        if not recent.empty:
            last_ret=recent["price_change"].iloc[-1]
            if last_ret>=0.02:
                bucket=5
            elif last_ret>=0.01:
                bucket=4
            elif last_ret>=-0.01:
                bucket=3
            elif last_ret>=-0.02:
                bucket=2
            else:
                bucket=1
            pv_score=(bucket-1)/4.0

    health_gauge=(0.25*oi_score+0.35*cot_score+0.40*pv_score)*10
    return round(health_gauge,2)

# --- Signal Generation ---
def generate_signal(health_gauge:float,buy_threshold:float=0.6,sell_threshold:float=-0.6)->str:
    if np.isnan(health_gauge):
        return "HOLD"
    if health_gauge>=buy_threshold*10:
        return "BUY"
    elif health_gauge<=sell_threshold*10:
        return "SELL"
    return "HOLD"

# --- Position Sizing ---
def calculate_lot_size(balance:float,price:float,asset_class:str,symbol:str,risk_per_trade:float=0.01)->float:
    contract_size=get_contract_size(asset_class,symbol)
    risk_amount=balance*risk_per_trade
    pip_value=contract_size*0.0001
    if pip_value==0:
        return 0
    lot_size=risk_amount/(pip_value*price)
    return round(lot_size,2)

# --- Backtest Logic ---
def process_assets(selected_assets:list,balance:float,leverage:float,buy_threshold:float,sell_threshold:float)->dict:
    results={}
    for asset in selected_assets:
        cot_df=fetch_cot_data(asset)
        ticker=assets[asset]
        price_df=fetch_price_data(ticker)
        merged_df=merge_cot_price(cot_df,price_df)
        health_gauge=calculate_health_gauge(merged_df)
        signal=generate_signal(health_gauge,buy_threshold,sell_threshold)
        asset_class,symbol=asset_classes.get(asset,("FX","default"))
        current_price=price_df["close"].iloc[-1] if not price_df.empty else 1
        lot_size=calculate_lot_size(balance,current_price,asset_class,symbol,0.01)
        results[asset]={"health_gauge":health_gauge,"signal":signal,"lot_size":lot_size,"current_price":current_price}
    return results

# --- Streamlit UI ---
def main():
    display_prop_firm_header()
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        selected_assets=st.multiselect("Select Assets",list(assets.keys()),default=list(assets.keys())[:3])
        years_back=st.slider("Years of Historical Data",min_value=1,max_value=10,value=3)
        leverage=st.number_input("Leverage",min_value=1.0,value=10.0)
        starting_balance=st.number_input("Account Balance",min_value=1000.0,value=10000.0)
        buy_threshold=st.slider("Buy Threshold",min_value=0.0,max_value=1.0,value=0.6)
        sell_threshold=st.slider("Sell Threshold",min_value=1.0,max_value=10.0,value=0.4)

    if st.button("🚀 Run Backtest"):
        with st.spinner("Fetching and processing data..."):
            results=process_assets(selected_assets,starting_balance,leverage,buy_threshold,sell_threshold)
        st.subheader("📊 Trading Signals & Positions")
        results_df=pd.DataFrame.from_dict(results,orient="index").reset_index().rename(columns={"index":"Asset"})
        st.dataframe(results_df)
        st.subheader("🎯 Health Gauge Analysis")
        fig=go.Figure()
        for asset in selected_assets:
            if results[asset]["health_gauge"] is not None:
                fig.add_trace(go.Bar(x=[asset],y=[results[asset]["health_gauge"]],name=asset,
                                     text=f"Signal: {results[asset]['signal']}\nLots: {results[asset]['lot_size']}",textposition="auto"))
        fig.update_layout(yaxis_title="Health Gauge Score",title="Health Gauge & Trading Signals",height=500)
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("💰 Account Summary")
        total_value=starting_balance
        for asset in selected_assets:
            total_value+=results[asset]["current_price"]*results[asset]["lot_size"]
        st.metric("Estimated Portfolio Value",f"${total_value:,.2f}")

if __name__=="__main__":
    main()