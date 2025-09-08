# chunk 1/3
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


# chunk 2/3

# --- Socrata Client ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# --- Data Fetching Functions (timezone-aware) ---
def fetch_cot_data(market_name:str,max_attempts:int=3)->pd.DataFrame:
    where_clause=f'market_and_exchange_names="{market_name}"'
    attempt=0
    while attempt<max_attempts:
        try:
            results=client.get("6dca-aqww",where=where_clause,order="report_date_as_yyyy_mm_dd DESC",limit=1500)
            if not results: return pd.DataFrame()
            df=pd.DataFrame.from_records(results)
            df["report_date"]=pd.to_datetime(df.get("report_date_as_yyyy_mm_dd"),errors="coerce").dt.tz_localize(None)
            df["report_date"]=pd.to_datetime(df["report_date"], utc=True)
            df["commercial_net"]=pd.to_numeric(df.get("commercial_long_all",0),errors="coerce")-pd.to_numeric(df.get("commercial_short_all",0),errors="coerce")
            df["non_commercial_net"]=pd.to_numeric(df.get("non_commercial_long_all",0),errors="coerce")-pd.to_numeric(df.get("non_commercial_short_all",0),errors="coerce")
            df["open_interest_all"]=pd.to_numeric(df.get("open_interest_all",np.nan),errors="coerce")
            return df.sort_values("report_date").reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error fetching COT for {market_name}: {e}")
            attempt+=1
            time.sleep(1)
    return pd.DataFrame()

def fetch_price_data(ticker:str,years:int=3)->pd.DataFrame:
    try:
        ticker_obj=Ticker(ticker)
        start_date=(datetime.datetime.now(pytz.UTC)-datetime.timedelta(days=365*years)).strftime("%Y-%m-%d")
        df=ticker_obj.history(start=start_date)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.index,pd.MultiIndex):
            df=df.reset_index(level=0,drop=True)
        df=df.reset_index().rename(columns={"date":"Date"})
        df["Date"]=pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df["Date"]=pd.to_datetime(df["Date"], utc=True)
        if "volume" not in df.columns: df["volume"]=np.nan
        for col in ["close","open","high","low"]:
            if col in df.columns:
                df[col]=pd.to_numeric(df[col],errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

def merge_cot_price(cot_df:pd.DataFrame,price_df:pd.DataFrame)->pd.DataFrame:
    if cot_df.empty or price_df.empty: return pd.DataFrame()
    cot_df = cot_df.copy()
    price_df = price_df.copy()
    cot_df["report_date"] = pd.to_datetime(cot_df["report_date"], utc=True)
    price_df["Date"] = pd.to_datetime(price_df["Date"], utc=True)
    merged = pd.merge_asof(price_df.sort_values("Date"), cot_df.sort_values("report_date"),
                           left_on="Date", right_on="report_date", direction="backward")
    for col in ["open_interest_all","commercial_net","non_commercial_net"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
        else:
            merged[col] = np.nan
    return merged

# --- Relative Volume ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if "volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    rolling_avg = df["volume"].rolling(window=window, min_periods=1).mean()
    return df["volume"] / rolling_avg

# --- Health Gauge (single snapshot) ---
def calculate_health_gauge(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("nan")
    # Open Interest score
    oi_score = 0.5
    if "open_interest_all" in df.columns:
        oi_series = df["open_interest_all"].dropna()
        if len(oi_series) > 1 and oi_series.max() != oi_series.min():
            oi_score = (oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min())
    # COT score
    cot_score = 0.5
    if "commercial_net" in df.columns and "non_commercial_net" in df.columns:
        short_term = df.tail(36)
        long_term = df.tail(144)
        st_score = 0.5
        lt_score = 0.5
        if not short_term["commercial_net"].isna().all() and short_term["commercial_net"].max() != short_term["commercial_net"].min():
            st_score = (short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / \
                       (short_term["commercial_net"].max() - short_term["commercial_net"].min())
        if not long_term["non_commercial_net"].isna().all() and long_term["non_commercial_net"].max() != long_term["non_commercial_net"].min():
            lt_score = (long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / \
                       (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min())
        cot_score = 0.4 * st_score + 0.6 * lt_score
    # Price/Volume score
    pv_score = 0.5
    if "close" in df.columns and "volume" in df.columns:
        df_loc = df.copy()
        df_loc["rvol"] = calculate_rvol(df_loc)
        recent = df_loc.tail(60)
        if not recent.empty and "close" in recent.columns:
            recent["price_change"] = recent["close"].pct_change()
            last_ret = recent["price_change"].iloc[-1] if len(recent) > 1 else 0.0
            if last_ret >= 0.02: bucket = 5
            elif last_ret >= 0.01: bucket = 4
            elif last_ret >= -0.01: bucket = 3
            elif last_ret >= -0.02: bucket = 2
            else: bucket = 1
            pv_score = (bucket - 1) / 4.0
    health_gauge = (0.25 * oi_score + 0.35 * cot_score + 0.40 * pv_score) * 10.0
    return float(np.round(health_gauge, 2))

# --- Produce Health Gauge time-series & signals ---
def produce_hg_and_signals(merged_df: pd.DataFrame, buy_threshold_input: int, sell_threshold_input: int) -> pd.DataFrame:
    if merged_df is None or merged_df.empty:
        return merged_df
    df = merged_df.copy().reset_index(drop=True)
    hgs = []
    buy_th = float(buy_threshold_input)
    sell_th = float(sell_threshold_input)
    for i in range(len(df)):
        slice_df = df.iloc[:i+1]
        hg = calculate_health_gauge(slice_df)
        hgs.append(hg)
    df["hg"] = hgs
    df["signal"] = "HOLD"
    df.loc[df["hg"] >= buy_th, "signal"] = "BUY"
    df.loc[df["hg"] <= sell_th, "signal"] = "SELL"
    return df


# chunk 3/3

# --- Base lot size calculation ---
def calculate_base_lot_size(account_balance: float, price: float, asset_class: str, symbol: str, risk_pct: float) -> float:
    contract_size = get_contract_size(asset_class, symbol)
    pip_val_per_unit = 0.0001 if asset_class == "FX" else 0.01
    assumed_stop_pips = 50
    loss_per_lot = contract_size * pip_val_per_unit * assumed_stop_pips
    if loss_per_lot <= 0 or np.isnan(loss_per_lot):
        return 0.0
    risk_amount = account_balance * risk_pct
    base_lots = risk_amount / loss_per_lot
    return float(max(0.0, base_lots))

# --- Backtest executor for a single asset ---
def execute_backtest_for_asset(merged_df: pd.DataFrame,
                               starting_balance: float,
                               leverage: float,
                               lot_multiplier: float,
                               buy_threshold_input: int,
                               sell_threshold_input: int,
                               risk_profile: dict):
    if merged_df is None or merged_df.empty:
        return pd.DataFrame(), pd.DataFrame(), starting_balance
    df = produce_hg_and_signals(merged_df, buy_threshold_input, sell_threshold_input).reset_index(drop=True)
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df.index)
    if "close" not in df.columns:
        logger.warning("No close price found; backtest will not run for this asset.")
        return pd.DataFrame(), pd.DataFrame(), starting_balance

    cash = float(starting_balance)
    equity_curve = []
    trades = []
    position = None
    peak_equity = cash
    max_drawdown = 0.0
    risk_per_trade = risk_profile.get("max_risk_per_trade", 0.01)
    profile_max_lot = risk_profile.get("max_lot_size", None)

    for i in range(len(df)):
        row = df.iloc[i]
        date = row["Date"]
        price = float(row["close"]) if not pd.isna(row["close"]) else np.nan
        signal = row["signal"]

        unreal = 0.0
        if position is not None and not np.isnan(price):
            direction = 1 if position["direction"] == "LONG" else -1
            unreal = (price - position["entry_price"]) * position["lots"] * get_contract_size(position["asset_class"], position["symbol"]) * direction

        equity = cash + unreal
        equity_curve.append({"date": date, "equity": equity, "cash": cash, "unrealized": unreal})

        if position is None:
            if signal in ["BUY", "SELL"]:
                asset_class, symbol = df.loc[i, "asset_class"] if "asset_class" in df.columns else ("FX", "default"), df.loc[i, "symbol"] if "symbol" in df.columns else "default"
                base_lots = calculate_base_lot_size(cash, price, asset_class, symbol, risk_per_trade)
                lots = base_lots * lot_multiplier
                if profile_max_lot is not None:
                    lots = min(lots, profile_max_lot)
                position_value = price * get_contract_size(asset_class, symbol) * lots
                required_margin = position_value / leverage if leverage > 0 else position_value
                if required_margin > cash:
                    scale = cash / required_margin if required_margin > 0 else 0.0
                    lots = lots * scale
                lots = float(max(0.0, lots))
                if lots > 0 and not np.isnan(price):
                    direction = "LONG" if signal == "BUY" else "SHORT"
                    entry_price = price
                    position = {
                        "direction": direction,
                        "entry_price": entry_price,
                        "lots": lots,
                        "entry_index": i,
                        "asset_class": asset_class,
                        "symbol": symbol
                    }
                    trades.append({
                        "entry_date": date,
                        "direction": direction,
                        "entry_price": entry_price,
                        "lots": lots,
                        "exit_date": None,
                        "exit_price": None,
                        "pnl": None
                    })
        else:
            if (position["direction"] == "LONG" and signal == "SELL") or (position["direction"] == "SHORT" and signal == "BUY"):
                exit_price = price
                direction = 1 if position["direction"] == "LONG" else -1
                contract_size = get_contract_size(position["asset_class"], position["symbol"])
                realized_pnl = (exit_price - position["entry_price"]) * position["lots"] * contract_size * direction
                cash += realized_pnl
                last_trade = trades[-1]
                last_trade["exit_date"] = date
                last_trade["exit_price"] = exit_price
                last_trade["pnl"] = realized_pnl
                position = None

        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    metrics = {
        "final_cash": cash,
        "final_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else cash,
        "max_drawdown": max_drawdown
    }
    return equity_df, trades_df, metrics

# --- Multi-asset backtest orchestration ---
def run_multi_asset_backtest(selected_assets, starting_balance, leverage, lot_multiplier,
                             buy_input, sell_input, risk_profile_name):
    risk_profile = RISK_PROFILES.get(risk_profile_name, {"max_risk_per_trade": 0.01})
    account_cash = float(starting_balance)
    all_equity_series = []
    all_trades = []

    for asset in selected_assets:
        ticker = assets[asset]
        cot_df = fetch_cot_data(asset)
        price_df = fetch_price_data(ticker)
        merged = merge_cot_price(cot_df, price_df)
        if merged.empty:
            logger.warning(f"No merged data for {asset}, skipping.")
            continue
        ac, sym = asset_classes.get(asset, ("FX", "default"))
        merged["asset_class"] = ac
        merged["symbol"] = sym

        equity_df, trades_df, metrics = execute_backtest_for_asset(
            merged,
            starting_balance=account_cash,
            leverage=leverage,
            lot_multiplier=lot_multiplier,
            buy_threshold_input=buy_input,
            sell_threshold_input=sell_input,
            risk_profile=risk_profile
        )

        if metrics and "final_cash" in metrics:
            account_cash = metrics["final_cash"]

        if not trades_df.empty:
            trades_df["asset"] = asset
            all_trades.append(trades_df)
        if not equity_df.empty:
            equity_df["asset"] = asset
            all_equity_series.append(equity_df)

    trades_combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_combined = pd.concat(all_equity_series, ignore_index=True) if all_equity_series else pd.DataFrame()

    if not equity_combined.empty:
        equity_by_date = equity_combined.groupby("date")["equity"].sum().sort_index()
        account_equity_df = equity_by_date.reset_index().rename(columns={"equity": "equity"})
        if account_equity_df["equity"].iloc[0] == 0:
            account_equity_df["equity"] = account_equity_df["equity"] + starting_balance
    else:
        account_equity_df = pd.DataFrame([{"date": pd.Timestamp.now(tz=pytz.UTC), "equity": account_cash}])

    return account_equity_df, trades_combined, account_cash


# --- Streamlit UI ---
def main():
    st.title("📊 Multi-Asset COT Health Gauge Backtester")
    with st.sidebar:
        st.header("⚙️ Backtest Settings")
        risk_profile_name = st.selectbox("Risk Profile", list(RISK_PROFILES.keys()), index=0)
        selected_assets = st.multiselect("Select Assets", list(assets.keys()), default=list(assets.keys())[:2])
        years_back = st.slider("Years of Historical Data", min_value=1, max_value=10, value=3)
        leverage = st.number_input("Leverage", min_value=1.0, value=10.0)
        starting_balance = st.number_input("Account Balance", min_value=1000.0, value=10000.0)
        buy_input = st.slider("Buy Threshold (1–10)", min_value=1, max_value=10, value=7)
        sell_input = st.slider("Sell Threshold (1–10)", min_value=1, max_value=10, value=3)
        lot_multiplier = st.number_input("Lot Size Multiplier", min_value=0.01, value=1.0, step=0.01)

    if st.button("🚀 Run Backtest"):
        with st.spinner("Running backtest across selected assets..."):
            account_equity_df, trades_df, final_cash = run_multi_asset_backtest(
                selected_assets=selected_assets,
                starting_balance=starting_balance,
                leverage=leverage,
                lot_multiplier=lot_multiplier,
                buy_input=buy_input,
                sell_input=sell_input,
                risk_profile_name=risk_profile_name
            )

        st.subheader("📈 Account Equity Curve")
        if not account_equity_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=account_equity_df["date"], y=account_equity_df["equity"], mode="lines", name="Equity"))
            fig.update_layout(title="Account Equity Curve", xaxis_title="Date", yaxis_title="Equity")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data available.")

        st.subheader("📊 Trades Executed")
        if not trades_df.empty:
            trades_display = trades_df.copy()
            trades_display["entry_date"] = pd.to_datetime(trades_display["entry_date"])
            if "exit_date" in trades_display:
                trades_display["exit_date"] = pd.to_datetime(trades_display["exit_date"])
            st.dataframe(trades_display.sort_values("entry_date", ascending=False).reset_index(drop=True))
        else:
            st.info("No trades executed.")

# --- Run the Streamlit app ---
if __name__ == "__main__":
    main()