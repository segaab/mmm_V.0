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

# --- Data Fetching Functions ---
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
                
            if "report_date_as_yyyy_mm_dd" in df.columns:  
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")  
            else:  
                logger.warning(f"Missing report_date_as_yyyy_mm_dd column in COT data for {market_name}")  
                df["report_date"] = pd.NaT  
                  
            if "commercial_long_all" in df.columns and "commercial_short_all" in df.columns:  
                df["commercial_long_all"] = pd.to_numeric(df["commercial_long_all"], errors="coerce")  
                df["commercial_short_all"] = pd.to_numeric(df["commercial_short_all"], errors="coerce")  
                df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]  
            else:  
                logger.warning(f"Missing commercial long/short columns in COT data for {market_name}")  
                df["commercial_net"] = np.nan  
                  
            if "non_commercial_long_all" in df.columns and "non_commercial_short_all" in df.columns:  
                df["non_commercial_long_all"] = pd.to_numeric(df["non_commercial_long_all"], errors="coerce")  
                df["non_commercial_short_all"] = pd.to_numeric(df["non_commercial_short_all"], errors="coerce")  
                df["non_commercial_net"] = df["non_commercial_long_all"] - df["non_commercial_short_all"]  
            else:  
                logger.warning(f"Missing non-commercial long/short columns in COT data for {market_name}")  
                df["non_commercial_net"] = np.nan  
                  
            if "open_interest_all" in df.columns:  
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")  
            else:  
                logger.warning(f"Missing open_interest_all column in COT data for {market_name}")  
                df["open_interest_all"] = np.nan  
                  
            return df.sort_values("report_date").reset_index(drop=True)  
        except Exception as e:  
            logger.error(f"Error fetching COT for {market_name}: {e}")  
            attempt += 1  
            time.sleep(1)  
            
    logger.error(f"Failed to fetch COT for {market_name} after {max_attempts} attempts.")  
    return pd.DataFrame()

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

def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if df is None or df.empty or "volume" not in df.columns:
        df["rvol"] = np.nan
        return df
    df = df.copy()
    df["rvol"] = df["volume"] / df["volume"].rolling(window, min_periods=1).mean()
    return df

def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        logger.warning("Empty COT or price data, cannot merge")
        return pd.DataFrame()

    required_cot_cols = ["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]  
    for col in required_cot_cols:  
        if col not in cot_df.columns:  
            logger.warning(f"Missing required column {col} in COT data")  
            cot_df[col] = np.nan  
          
    cot_small = cot_df[required_cot_cols].copy()  
    cot_small.rename(columns={"report_date": "date"}, inplace=True)  

    price_df["date"] = pd.to_datetime(price_df["date"])  
    cot_small["date"] = pd.to_datetime(cot_small["date"])  

    merged = pd.merge_asof(  
        price_df.sort_values("date"),  
        cot_small.sort_values("date"),  
        on="date",  
        direction="backward"  
    )  

    for col in ["open_interest_all", "commercial_net", "non_commercial_net"]:  
        merged[col] = merged[col].ffill()  
          
    return merged

def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return float("nan")

    if "report_date" not in cot_df.columns and "date" in cot_df.columns:  
        cot_df = cot_df.rename(columns={"date": "report_date"})  

    if "commercial_net" not in cot_df.columns:  
        logger.warning("commercial_net column missing in COT data")  
        return float("nan")  
    if "non_commercial_net" not in cot_df.columns:  
        logger.warning("non_commercial_net column missing in COT data")  
        return float("nan")  
    if "open_interest_all" not in price_df.columns:  
        logger.warning("open_interest_all column missing in merged data")  
        return float("nan")  

    df = price_df.copy()  
    df["rvol"] = df.get("rvol", np.nan)  
    last_date = df["date"].max()  
    one_year_ago = last_date - pd.Timedelta(days=365)  
    three_months_ago = last_date - pd.Timedelta(days=90)  

    oi_series = df["open_interest_all"].dropna()  
    oi_score = float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9)) if not oi_series.empty else 0.0  

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

def generate_signals(df, buy_threshold=0.3, sell_threshold=0.7):
    if df is None or df.empty:
        return pd.DataFrame()

    health_gauges = []  
    for i in range(len(df)):  
        date = df.iloc[i]["date"]  
        cot_subset = df[df["date"] <= date].copy()  
        price_subset = df[df["date"] <= date].copy()  
          
        if not cot_subset.empty and not price_subset.empty and "commercial_net" in df.columns and "non_commercial_net" in df.columns:  
            hg = calculate_health_gauge(cot_subset, price_subset)  
        else:  
            hg = np.nan  
        health_gauges.append(hg)  

    df = df.copy()  
    df["hg"] = health_gauges  
    df["hg"] = df["hg"].fillna(0).clip(0, 10) / 10  

    df["signal"] = 0  
    df.loc[df["hg"] > sell_threshold, "signal"] = -1  
    df.loc[df["hg"] < buy_threshold, "signal"] = 1  

    return df

# --- Enhanced Position Size Calculation with Risk Profile ---
def calculate_position_size_with_profile(account_balance, risk_profile, asset_name, price, stop_loss_pips=50):
    try:
        asset_class, symbol = asset_classes.get(asset_name, ("FX", "default"))
    except:
        logger.warning(f"Unknown asset: {asset_name}, using FX default")
        asset_class, symbol = "FX", "default"

    # Get leverage from risk profile  
    max_leverage = risk_profile["leverage_limits"].get(asset_class, 10)  
    
    # Maximum risk per trade from profile  
    max_risk_per_trade = risk_profile.get("max_risk_per_trade", 0.02)  
    
    contract_size = get_contract_size(asset_class, symbol)  
    pip_size = 0.01 if asset_class != "FX" else 0.0001  
    
    # Calculate maximum position size based on risk  
    max_risk_amount = account_balance * max_risk_per_trade  
    loss_per_lot = contract_size * pip_size * stop_loss_pips  
    max_lots = max_risk_amount / loss_per_lot if loss_per_lot > 0 else 0  
    
    # Apply leverage limits  
    position_value_per_lot = contract_size * price  
    max_lots_leverage = (account_balance * max_leverage) / position_value_per_lot  
    
    # Take minimum of risk and leverage constraints  
    final_lots = min(max_lots, max_lots_leverage)  
    
    # Apply TopOneTrader specific lot limits  
    if "max_lot_size" in risk_profile:  
        final_lots = min(final_lots, risk_profile["max_lot_size"])  
    
    return max(0, final_lots)

# --- Enhanced Backtest with Risk Profile ---
def execute_backtest_with_profile(signals_df: pd.DataFrame, asset_name: str, risk_profile: dict,
starting_balance=10000, lot_size=1.0, stop_loss_pips=50):
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    balance = starting_balance  
    equity_curve = []  
    trades = []  
    
    # Challenge tracking  
    current_phase = 1  
    phase_start_balance = starting_balance  
    daily_starting_balance = starting_balance  
    current_date = None  
    max_drawdown = 0  
    peak_balance = starting_balance  
    trading_days = 0  
    daily_profit = 0  
    total_phase_profit = 0  
    
    # Challenge targets  
    phase_targets = [  
        risk_profile.get("phase_1_target", 0.08),  
        risk_profile.get("phase_2_target", 0.05),  
        risk_profile.get("phase_3_target", None)  
    ]  
    
    challenge_status = "Phase 1"  
    is_funded = False  

    for i in range(1, len(signals_df)):  
        current_row = signals_df.iloc[i]  
          
        # Check for new trading day  
        if current_row["date"].date() != current_date:  
            if current_date is not None:  
                trading_days += 1  
            daily_starting_balance = balance  
            daily_profit = 0  
            current_date = current_row["date"].date()  
              
        # Daily loss limit check  
        daily_loss_pct = (daily_starting_balance - balance) / daily_starting_balance  
        if daily_loss_pct >= risk_profile["daily_drawdown_limit"]:  
            challenge_status = "FAILED - Daily Loss Limit Exceeded"  
            break  
              
        # Maximum drawdown check  
        if balance > peak_balance:  
            peak_balance = balance  
        current_drawdown = (peak_balance - balance) / peak_balance  
        if current_drawdown > max_drawdown:  
            max_drawdown = current_drawdown  
        if current_drawdown >= risk_profile["max_drawdown_limit"]:  
            challenge_status = "FAILED - Maximum Drawdown Exceeded"  
            break  

        # Check phase completion  
        phase_profit_pct = (balance - phase_start_balance) / phase_start_balance  
        current_target = phase_targets[current_phase - 1] if current_phase <= len(phase_targets) else None  
          
        if current_target and phase_profit_pct >= current_target:  
            if current_phase == len([t for t in phase_targets if t is not None]):  
                challenge_status = "FUNDED"  
                is_funded = True  
            else:  
                current_phase += 1  
                phase_start_balance = balance  
                challenge_status = f"Phase {current_phase}"  

        signal = signals_df.iloc[i-1].get("signal", 0)  
        price_open = current_row.get("close", np.nan)  
        price_prev = signals_df.iloc[i-1].get("close", np.nan)  

        if signal != 0 and not np.isnan(price_open) and not np.isnan(price_prev):  
            # Calculate position size using risk profile  
            position_lots = calculate_position_size_with_profile(  
                account_balance=balance,  
                risk_profile=risk_profile,  
                asset_name=asset_name,  
                price=price_open,  
                stop_loss_pips=stop_loss_pips  
            ) * lot_size  

            if position_lots > 0:  
                # Calculate trade return  
                price_change_pct = (price_open - price_prev) / price_prev  
                leverage = risk_profile["leverage_limits"].get(asset_classes.get(asset_name, ("FX", "default"))[0], 10)  
                trade_return_pct = price_change_pct * leverage * signal * position_lots * 0.01  
                  
                # Apply consistency rule for TopOneTrader  
                if "consistency_rule" in risk_profile:  
                    max_daily_target = current_target * risk_profile["consistency_rule"] if current_target else 0.05  
                    if abs(trade_return_pct) > max_daily_target:  
                        trade_return_pct = max_daily_target if trade_return_pct > 0 else -max_daily_target  
                  
                trade_return_amount = balance * trade_return_pct  
                new_balance = balance + trade_return_amount  
                  
                # Check if trade would breach daily limit  
                new_daily_loss = (daily_starting_balance - new_balance) / daily_starting_balance  
                if new_daily_loss < risk_profile["daily_drawdown_limit"]:  
                    balance = new_balance  
                    daily_profit += trade_return_amount  
                    total_phase_profit += trade_return_amount  
                      
                    trades.append({  
                        "date": current_row["date"],  
                        "signal": signal,  
                        "price": price_open,  
                        "position_lots": position_lots,  
                        "trade_return_pct": trade_return_pct,  
                        "trade_return_amount": trade_return_amount,  
                        "balance": balance,  
                        "phase": current_phase,  
                        "drawdown": current_drawdown,  
                        "daily_loss": daily_loss_pct,  
                        "phase_progress": phase_profit_pct  
                    })  

        equity_curve.append({  
            "date": current_row["date"],   
            "balance": balance,  
            "drawdown": current_drawdown,  
            "daily_loss": daily_loss_pct,  
            "phase": current_phase,  
            "phase_progress": phase_profit_pct,  
            "challenge_status": challenge_status,  
            "target": current_target  
        })  

    equity_df = pd.DataFrame(equity_curve)  
    trades_df = pd.DataFrame(trades)  

    # Calculate performance metrics  
    wins = trades_df[trades_df["trade_return_amount"] > 0].shape[0] if not trades_df.empty else 0  
    losses = trades_df[trades_df["trade_return_amount"] <= 0].shape[0] if not trades_df.empty else 0  
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0  

    total_return = (balance - starting_balance) / starting_balance  

    metrics = {  
        "final_balance": balance,  
        "total_return": total_return,  
        "total_return_pct": total_return * 100,  
        "num_trades": len(trades_df),  
        "wins": wins,  
        "losses": losses,  
        "win_rate": win_rate * 100,  
        "max_drawdown": max_drawdown * 100,  
        "max_daily_loss": equity_df["daily_loss"].max() * 100 if not equity_df.empty else 0,  
        "trading_days": trading_days,  
        "current_phase": current_phase,  
        "challenge_status": challenge_status,  
        "is_funded": is_funded,  
        "phase_progress": equity_df["phase_progress"].iloc[-1] * 100 if not equity_df.empty else 0  
    }  

    return equity_df, trades_df, metrics

# --- Main Application ---
def main():
    display_prop_firm_header()

    # Sidebar for parameters  
    with st.sidebar:  
        st.header("🎯 Challenge Parameters")  
          
        # Risk Profile Selection  
        selected_profile = st.selectbox(  
            "Select Prop Firm Challenge",  
            list(RISK_PROFILES.keys()),  
            index=0  
        )  
        risk_profile = RISK_PROFILES[selected_profile]  
          
        st.markdown("---")  
          
        # Display selected profile info  
        st.markdown(f"""  
        **{risk_profile['name']} Rules:**  
        - Daily DD Limit: {risk_profile['daily_drawdown_limit']*100:.0f}%  
        - Max DD Limit: {risk_profile['max_drawdown_limit']*100:.0f}%  
        - Phase 1 Target: {risk_profile['phase_1_target']*100:.0f}%  
        - Phase 2 Target: {risk_profile['phase_2_target']*100:.0f}%  
        """)  
          
        if risk_profile.get('phase_3_target'):  
            st.markdown(f"- Phase 3 Target: {risk_profile['phase_3_target']*100:.0f}%")  
          
        st.markdown("---")  

        # Trading Parameters  
        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)  
        years_back = st.slider("Years to Backtest", min_value=1, max_value=10, value=3)  
          
        st.markdown("**Strategy Parameters**")  
        buy_thresh = st.number_input("Buy Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)  
        sell_thresh = st.number_input("Sell Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)  
          
        st.markdown("**Risk Parameters**")  
        lot_size = st.number_input("Lot Size Multiplier", min_value=0.01, value=1.0, step=0.01)  
        stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1, value=50, step=1)  
        starting_balance = st.number_input("Starting Balance", min_value=1000, value=10000, step=1000)  

    # Main content area  
    with st.spinner("🔄 Loading and processing market data..."):  
        # Load and process data  
        price_df = load_price_data(selected_asset, years_back)  
        cot_df = load_cot_data(selected_asset)  

        if price_df.empty or cot_df.empty:  
            st.error(f"❌ No data available for {selected_asset}. Please try another asset.")  
            return  

        price_df = calculate_rvol(price_df)  
        signals_df = merge_cot_price(cot_df, price_df)  
        signals_df = generate_signals(signals_df, buy_threshold=buy_thresh, sell_threshold=sell_thresh)  

    # Execute backtest with risk profile  
    equity_df, trades_df, metrics = execute_backtest_with_profile(  
        signals_df,  
        asset_name=selected_asset,  
        risk_profile=risk_profile,  
        starting_balance=starting_balance,  
        lot_size=lot_size,  
        stop_loss_pips=stop_loss_pips  
    )  

    # Display challenge status  
    display_challenge_status(metrics, risk_profile)  
    
    # Display metrics dashboard  
    create_prop_metrics_dashboard(metrics, risk_profile)  
    
    # Performance charts  
    st.subheader("📊 Performance Analysis")  
    create_phase_progress_chart(equity_df, risk_profile)  
    
    # Detailed metrics  
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.subheader("📈 Challenge Metrics")  
        if metrics["is_funded"]:  
            st.success("🎉 CHALLENGE PASSED - FUNDED!")  
        elif "FAILED" in metrics["challenge_status"]:  
            st.error(f"❌ {metrics['challenge_status']}")  
        else:  
            st.info(f"⏳ In Progress - {metrics['challenge_status']}")  
              
        st.metric("Current Phase", metrics["current_phase"])  
        st.metric("Phase Progress", f"{metrics['phase_progress']:.2f}%")  
        st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")  
        st.metric("Max Daily Loss", f"{metrics['max_daily_loss']:.2f}%")  
    
    with col2:  
        st.subheader("📊 Trading Statistics")  
        st.metric("Total Trades", metrics["num_trades"])  
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")  
        st.metric("Wins", metrics["wins"])  
        st.metric("Losses", metrics["losses"])  
        st.metric("Trading Days", metrics["trading_days"])  

    # Health Gauge Chart  
    if not signals_df.empty and "hg" in signals_df.columns:  
        st.subheader("🎯 Health Gauge Analysis")  
        fig = go.Figure()  
          
        fig.add_trace(go.Scatter(  
            x=signals_df["date"],  
            y=signals_df["hg"],  
            mode="lines",  
            name="Health Gauge",  
            line=dict(color="#667eea", width=2)  
        ))  
          
        fig.add_hline(y=buy_thresh, line_dash="dash", line_color="#2ed573",   
                     annotation_text=f"Buy Threshold ({buy_thresh})")  
        fig.add_hline(y=sell_thresh, line_dash="dash", line_color="#ff4757",   
                     annotation_text=f"Sell Threshold ({sell_thresh})")  
          
        fig.update_layout(  
            title="Health Gauge Signal Analysis",  
            xaxis_title="Date",  
            yaxis_title="Health Gauge Value",  
            height=400  
        )  
          
        st.plotly_chart(fig, use_container_width=True)  

    # Trades table  
    if not trades_df.empty:  
        st.subheader("📋 Trade History")  
          
        # Format trades dataframe for display  
        display_trades = trades_df.copy()  
        display_trades["date"] = display_trades["date"].dt.strftime("%Y-%m-%d")  
        display_trades["trade_return_pct"] = display_trades["trade_return_pct"].round(4)  
        display_trades["balance"] = display_trades["balance"].round(2)  
        display_trades["phase_progress"] = display_trades["phase_progress"].round(4)  
          
        st.dataframe(  
            display_trades,  
            use_container_width=True,  
            column_config={  
                "trade_return_pct": st.column_config.NumberColumn(  
                    "Return %",  
                    format="%.4f"  
                ),  
                "balance": st.column_config.NumberColumn(  
                    "Balance",  
                    format="$%.2f"  
                ),  
                "phase_progress": st.column_config.NumberColumn(  
                    "Phase Progress",  
                    format="%.4f"  
                )  
            }  
        )  

    # Risk Profile Summary  
    with st.expander("📋 Risk Profile Details"):  
        st.json(risk_profile)

if __name__ == "__main__":
    main()
