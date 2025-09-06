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
    cot_results = {}
    price_results = {}
    lock = threading.Lock()
    items = list(assets_dict.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for i in range(0, len(batches), 2):
        active_threads = []
        for j in range(i, min(i + 2, len(batches))):
            t = threading.Thread(
                target=fetch_batch,
                args=(batches[j], start_date, end_date, cot_results, price_results, lock),
                daemon=True,
            )
            t.start()
            active_threads.append(t)
        for t in active_threads:
            t.join()
        time.sleep(0.5)
    return cot_results, price_results

# --- Health Gauge Function ---
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return np.nan
    last_date = pd.to_datetime(price_df["date"]).max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # Open Interest (25%)
    oi = price_df["open_interest_all"].dropna()
    oi_score = 0.0 if oi.empty else float(((oi - oi.min()) / (oi.max() - oi.min() + 1e-9)).iloc[-1])

    # COT Analytics (35%)
    commercial = cot_df[["report_date", "commercial_net"]].dropna().copy()
    commercial["report_date"] = pd.to_datetime(commercial["report_date"])
    short_term = commercial[commercial["report_date"] >= three_months_ago]

    noncomm = cot_df[["report_date", "non_commercial_net"]].dropna().copy()
    noncomm["report_date"] = pd.to_datetime(noncomm["report_date"])
    long_term = noncomm[noncomm["report_date"] >= one_year_ago]

    st_score = 0.0 if short_term.empty else float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9))
    lt_score = 0.0 if long_term.empty else float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) / (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9))
    cot_analytics = 0.4 * st_score + 0.6 * lt_score

    # Price Return + RVOL + Volume Spike (40%)
    recent = price_df[pd.to_datetime(price_df["date"]) >= three_months_ago].copy()
    if recent.empty or "rvol" not in recent.columns:
        pv_score = 0.0
    else:
        close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
        vol_col = "volume" if "volume" in recent.columns else ("Volume" if "Volume" in recent.columns else None)
        if close_col is None or vol_col is None:
            pv_score = 0.0
        else:
            recent["return"] = recent[close_col].pct_change().fillna(0.0)
            rvol_75 = recent["rvol"].quantile(0.75)
            recent["vol_avg20"] = recent[vol_col].rolling(20).mean()
            recent["vol_spike"] = recent[vol_col] > recent["vol_avg20"]
            filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
            if filt.empty:
                pv_score = 0.0
            else:
                last_ret = float(filt["return"].iloc[-1])
                if last_ret >= 0.02: bucket = 5
                elif 0.01 <= last_ret < 0.02: bucket = 4
                elif -0.01 <= last_ret < 0.01: bucket = 3
                elif -0.02 <= last_ret < -0.01: bucket = 2
                else: bucket = 1
                pv_score = (bucket - 1) / 4.0

    health_score = (0.25 * oi_score + 0.35 * cot_analytics + 0.40 * pv_score) * 10.0
    return float(health_score)

# --- Calculate Health Gauge Time Series ---
def calculate_health_gauge_series(cot_df, price_df):
    if cot_df.empty or price_df.empty:
        return pd.DataFrame({'date': [], 'health_gauge': []})
    
    health_scores = []
    dates = []
    
    for i in range(100, len(price_df)):
        date = price_df.iloc[i]['date']
        cot_slice = cot_df[cot_df['report_date'] <= date]
        price_slice = price_df.iloc[:i+1]
        
        if len(cot_slice) > 10 and len(price_slice) > 30:  # Ensure enough data
            health_score = calculate_health_gauge(cot_slice, price_slice)
            health_scores.append(health_score)
            dates.append(date)
    
    return pd.DataFrame({'date': dates, 'health_gauge': health_scores})

# --- Price Extension Bands ---
def calculate_price_bands(price_df, lookback=60):
    close_col = "close" if "close" in price_df.columns else "Close"
    high_col = "high" if "high" in price_df.columns else "High"
    low_col = "low" if "low" in price_df.columns else "Low"
    
    price_df = price_df.copy()
    price_df['upper_band'] = price_df[high_col].rolling(lookback).max()
    price_df['lower_band'] = price_df[low_col].rolling(lookback).min()
    price_df['mid_band'] = (price_df['upper_band'] + price_df['lower_band']) / 2
    
    # Calculate extensions
    price_df['range'] = price_df['upper_band'] - price_df['lower_band']
    price_df['extension'] = (price_df[close_col] - price_df['mid_band']) / (price_df['range'] / 2)
    
    return price_df

# --- Signal Generation ---
def generate_signals(health_df, price_df, buy_threshold, sell_threshold, rvol_threshold=1.5):
    if health_df.empty or price_df.empty:
        return pd.DataFrame()
    
    signals = pd.merge(health_df, price_df, on='date', how='inner')
    
    # Generate signals based on health gauge and thresholds
    signals['buy_signal'] = (signals['health_gauge'] >= buy_threshold) & (signals['rvol'] >= rvol_threshold)
    signals['sell_signal'] = (signals['health_gauge'] <= sell_threshold) & (signals['rvol'] >= rvol_threshold)
    
    # Add volatility and extension filters
    signals['volatility_high'] = signals['rvol'] > 2.0
    signals['extension_high'] = abs(signals['extension']) > 0.8
    
    # Strong signals when volatility and health gauge align
    signals['strong_buy'] = signals['buy_signal'] & signals['volatility_high'] & (signals['extension'] < 0)
    signals['strong_sell'] = signals['sell_signal'] & signals['volatility_high'] & (signals['extension'] > 0)
    
    # Add reversal warnings
    signals['buy_reversal_warning'] = signals['buy_signal'] & (signals['extension'] > 0.8)
    signals['sell_reversal_warning'] = signals['sell_signal'] & (signals['extension'] < -0.8)
    
    return signals

# --- Execute Backtest ---
def execute_backtest(signals_df, starting_balance=10000, leverage=15, position_size='medium'):
    if signals_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    close_col = "close" if "close" in signals_df.columns else "Close"
    
    # Define position size multiplier
    size_multipliers = {'small': 0.1, 'medium': 0.2, 'large': 0.3, 'heavy': 0.5}
    size_mult = size_multipliers.get(position_size, 0.2)
    
    # Set up results tracking
    results = []
    balance = starting_balance
    in_position = False
    position_type = None
    entry_price = 0
    entry_date = None
    trade_id = 0
    max_drawdown = 0
    
    # Iterate through each day
    for i, row in signals_df.iterrows():
        date = row['date']
        price = row[close_col]
        
        # If not in position, check for new entry
        if not in_position:
            if row['buy_signal']:
                # Enter long position
                trade_id += 1
                position_size_usd = balance * size_mult * leverage
                entry_price = price
                entry_date = date
                in_position = True
                position_type = 'long'
                results.append({
                    'date': date,
                    'balance': balance,
                    'action': 'buy_entry',
                    'price': price,
                    'trade_id': trade_id,
                    'position_type': position_type,
                    'drawdown': 0,
                    'r_multiple': 0
                })
            elif row['sell_signal']:
                # Enter short position
                trade_id += 1
                position_size_usd = balance * size_mult * leverage
                entry_price = price
                entry_date = date
                in_position = True
                position_type = 'short'
                results.append({
                    'date': date,
                    'balance': balance,
                    'action': 'sell_entry',
                    'price': price,
                    'trade_id': trade_id,
                    'position_type': position_type,
                    'drawdown': 0,
                    'r_multiple': 0
                })
        
        # If in position, check for exit
        else:
            # Calculate unrealized P&L
            if position_type == 'long':
                unrealized_pl = (price - entry_price) / entry_price * leverage * balance * size_mult
                r_multiple = (price - entry_price) / (entry_price * 0.01) * 0.2  # Using 1% as 1R
                
                # Track drawdown
                drawdown = min(0, unrealized_pl) / (balance * size_mult * leverage) * 100
                max_drawdown = min(max_drawdown, drawdown)
                
                # Exit conditions: opposite signal, take profit (3R), or stop loss (2R)
                if row['sell_signal'] or r_multiple >= 3 or r_multiple <= -2:
                    # Close position
                    realized_pl = unrealized_pl
                    balance += realized_pl
                    in_position = False
                    
                    exit_reason = 'signal_exit'
                    if r_multiple >= 3:
                        exit_reason = 'take_profit'
                    elif r_multiple <= -2:
                        exit_reason = 'stop_loss'
                    
                    results.append({
                        'date': date,
                        'balance': balance,
                        'action': f'long_exit_{exit_reason}',
                        'price': price,
                        'trade_id': trade_id,
                        'position_type': position_type,
                        'drawdown': drawdown,
                        'r_multiple': r_multiple
                    })
            
            elif position_type == 'short':
                unrealized_pl = (entry_price - price) / entry_price * leverage * balance * size_mult
                r_multiple = (entry_price - price) / (entry_price * 0.01) * 0.2  # Using 1% as 1R
                
                # Track drawdown
                drawdown = min(0, unrealized_pl) / (balance * size_mult * leverage) * 100
                max_drawdown = min(max_drawdown, drawdown)
                
                # Exit conditions: opposite signal, take profit (3R), or stop loss (2R)
                if row['buy_signal'] or r_multiple >= 3 or r_multiple <= -2:
                    # Close position
                    realized_pl = unrealized_pl
                    balance += realized_pl
                    in_position = False
                    
                    exit_reason = 'signal_exit'
                    if r_multiple >= 3:
                        exit_reason = 'take_profit'
                    elif r_multiple <= -2:
                        exit_reason = 'stop_loss'
                    
                    results.append({
                        'date': date,
                        'balance': balance,
                        'action': f'short_exit_{exit_reason}',
                        'price': price,
                        'trade_id': trade_id,
                        'position_type': position_type,
                        'drawdown': drawdown,
                        'r_multiple': r_multiple
                    })
    
    # Close any open position at the end
    if in_position:
        last_row = signals_df.iloc[-1]
        last_price = last_row[close_col]
        last_date = last_row['date']
        
        if position_type == 'long':
            unrealized_pl = (last_price - entry_price) / entry_price * leverage * balance * size_mult
            r_multiple = (last_price - entry_price) / (entry_price * 0.01) * 0.2
        else:
            unrealized_pl = (entry_price - last_price) / entry_price * leverage * balance * size_mult
            r_multiple = (entry_price - last_price) / (entry_price * 0.01) * 0.2
        
        drawdown = min(0, unrealized_pl) / (balance * size_mult * leverage) * 100
        balance += unrealized_pl
        
        results.append({
            'date': last_date,
            'balance': balance,
            'action': f'{position_type}_exit_end',
            'price': last_price,
            'trade_id': trade_id,
            'position_type': position_type,
            'drawdown': drawdown,
            'r_multiple': r_multiple
        })
    
    results_df = pd.DataFrame(results)
    
    # Fill in equity curve for all dates
    if not results_df.empty:
        equity_curve = pd.DataFrame({'date': signals_df['date']})
        equity_curve = pd.merge_asof(equity_curve.sort_values('date'), 
                                     results_df[['date', 'balance']].sort_values('date'), 
                                     on='date', 
                                     direction='backward')
        equity_curve['balance'] = equity_curve['balance'].fillna(method='ffill').fillna(starting_balance)
        
        # Calculate additional metrics
        if len(results_df) > 0:
            total_trades = results_df['trade_id'].nunique()
            profitable_trades = results_df[results_df['r_multiple'] > 0]['trade_id'].nunique()
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            avg_r = results_df[results_df['action'].str.contains('exit')]['r_multiple'].mean()
            max_r = results_df[results_df['action'].str.contains('exit')]['r_multiple'].max()
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_r': avg_r,
                'max_r': max_r,
                'max_drawdown': max_drawdown,
                'final_balance': balance,
                'roi': (balance - starting_balance) / starting_balance
            }
        else:
            metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_r': 0,
                'max_r': 0,
                'max_drawdown': 0,
                'final_balance': starting_balance,
                'roi': 0
            }
        
        return results_df, equity_curve, metrics
    
    else:
        # Return empty dataframes and default metrics if no trades
        return pd.DataFrame(), pd.DataFrame({'date': signals_df['date'], 'balance': starting_balance}), {
            'total_trades': 0, 
            'win_rate': 0,
            'avg_r': 0,
            'max_r': 0,
            'max_drawdown': 0,
            'final_balance': starting_balance,
            'roi': 0
        }

# --- Neural Network Model ---
class ThresholdOptimizer(nn.Module):
    def __init__(self, input_size):
        super(ThresholdOptimizer, self).__init__()
        # Simple feedforward neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()  # Output is between 0 and 1, scaled later
        )
    
    def forward(self, x):
        return self.model(x)

# --- Prepare Data for NN ---
def prepare_nn_data(health_df, price_df, lookback=20):
    if health_df.empty or price_df.empty:
        return None, None
    
    # Merge data
    data = pd.merge(health_df, price_df, on='date', how='inner')
    
    # Feature engineering
    close_col = "close" if "close" in data.columns else "Close"
    data['return'] = data[close_col].pct_change()
    data['volatility'] = data['return'].rolling(lookback).std()
    data['health_gauge_ma'] = data['health_gauge'].rolling(lookback).mean()
    data['health_gauge_std'] = data['health_gauge'].rolling(lookback).std()
    data['rvol_ma'] = data['rvol'].rolling(lookback).mean()
    
    # Add COT features
    data['commercial_net_change'] = data['commercial_net'].diff()
    data['non_commercial_net_change'] = data['non_commercial_net'].diff()
    
    # Drop NAs from feature creation
    data = data.dropna()
    
    if data.empty:
        return None, None
    
    # Select features and target
    features = [
        'health_gauge', 'health_gauge_ma', 'health_gauge_std',
        'rvol', 'rvol_ma', 'volatility', 'extension',
        'commercial_net', 'non_commercial_net',
        'commercial_net_change', 'non_commercial_net_change'
    ]
    
    # Prepare X data
    X = data[features].values
    
    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # No explicit Y - we'll use backtest performance as feedback
    return X_scaled, data['date'].values

# --- Train NN with Reinforcement Learning --- (FIXED)
def train_nn(X, dates, cot_df, price_df, epochs=20, lr=0.001, batch_size=32):
    if X is None or len(X) == 0:
        return None, None, None
    
    input_size = X.shape[1]
    model = ThresholdOptimizer(input_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Starting thresholds
    best_buy_threshold = 7.0
    best_sell_threshold = 3.0
    best_roi = -float('inf')
    
    # Training loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # First calculate the health gauge series once (FIX)
    health_gauge_series = calculate_health_gauge_series(cot_df, price_df)
    
    for epoch in range(epochs):
        status_text.text(f"Training epoch {epoch+1}/{epochs}")
        progress_bar.progress((epoch + 1) / epochs)
        
        # Forward pass to get thresholds
        outputs = model(torch.FloatTensor(X))
        buy_threshold = 5.0 + outputs[:, 0].mean().item() * 5.0  # Scale to 5-10
        sell_threshold = outputs[:, 1].mean().item() * 5.0  # Scale to 0-5
        
        # Create health gauge dataframe directly (FIX)
        health_df = health_gauge_series.copy()
        
        # Calculate price bands
        price_with_bands = calculate_price_bands(price_df)
        
        # Generate signals with current thresholds
        signals = generate_signals(health_df, price_with_bands, buy_threshold, sell_threshold)
        
        # Execute backtest
        _, _, metrics = execute_backtest(signals)
        
        roi = metrics['roi']
        avg_r = metrics['avg_r'] if 'avg_r' in metrics else 0
        win_rate = metrics['win_rate'] if 'win_rate' in metrics else 0
        drawdown = abs(metrics['max_drawdown']) if 'max_drawdown' in metrics else 0
        
        # Custom reward function
        reward = roi - 0.5 * drawdown + 0.3 * avg_r + 0.2 * win_rate
        
        # If this is the best performance so far, save the thresholds
        if reward > best_roi:
            best_roi = reward
            best_buy_threshold = buy_threshold
            best_sell_threshold = sell_threshold
        
        # Backward pass (optimize to maximize reward)
        loss = -torch.tensor(reward, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return model, best_buy_threshold, best_sell_threshold


def main():
    st.title("Health Gauge Trading Strategy Backtester")
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Backtest Parameters")
        
        selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)
        years_back = st.slider("Years to Backtest", 1, 10, 5)
        
        buy_threshold = st.slider("Buy Threshold (Health Gauge ≥)", 5.0, 10.0, 7.0, 0.1)
        sell_threshold = st.slider("Sell Threshold (Health Gauge ≤)", 0.0, 5.0, 3.0, 0.1)
        
        starting_balance = st.number_input("Starting Balance ($)", 1000, 1000000, 10000, 1000)
        leverage = st.slider("Leverage", 1, 30, 15)
        position_size = st.select_slider("Position Size", options=["small", "medium", "large", "heavy"], value="medium")
        
        st.header("Neural Network Settings")
        use_nn = st.checkbox("Use Neural Network Optimization", value=True)
        epochs = st.slider("Training Epochs", 5, 100, 20, 5, disabled=not use_nn)
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, disabled=not use_nn)
        
        run_backtest = st.button("Run Backtest")
    
    if run_backtest:
        with st.spinner("Fetching data and running backtest..."):
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=years_back * 365)
            
            cot_results, price_results = fetch_all_data({selected_asset: assets[selected_asset]}, start_date, end_date)
            cot_df = cot_results.get(selected_asset, pd.DataFrame())
            price_df = price_results.get(selected_asset, pd.DataFrame())
            
            if cot_df.empty or price_df.empty:
                st.error(f"Could not fetch data for {selected_asset}.")
                return
            
            health_df = calculate_health_gauge_series(cot_df, price_df)
            price_with_bands = calculate_price_bands(price_df)
            
            if use_nn:
                st.subheader("Neural Network Optimization")
                X, dates = prepare_nn_data(health_df, price_with_bands)
                
                if X is not None and len(X) > 0:
                    model, nn_buy_threshold, nn_sell_threshold = train_nn(X, dates, cot_df, price_df, epochs, learning_rate)
                    if model is not None:
                        st.success("Neural network training completed!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Buy Threshold", f"{buy_threshold:.1f}")
                            st.metric("NN Buy Threshold", f"{nn_buy_threshold:.1f}")
                        with col2:
                            st.metric("Original Sell Threshold", f"{sell_threshold:.1f}")
                            st.metric("NN Sell Threshold", f"{nn_sell_threshold:.1f}")
                        
                        use_optimized = st.checkbox("Use optimized thresholds for backtest", value=True)
                        if use_optimized:
                            buy_threshold = nn_buy_threshold
                            sell_threshold = nn_sell_threshold
                else:
                    st.warning("Not enough data for NN training. Using original thresholds.")
            
            signals = generate_signals(health_df, price_with_bands, buy_threshold, sell_threshold)
            trades_df, equity_curve, metrics = execute_backtest(signals, starting_balance, leverage, position_size)
            
            # Display results
            st.header("Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Balance", f"${metrics['final_balance']:.2f}")
                st.metric("ROI", f"{metrics['roi']*100:.2f}%")
            with col2:
                st.metric("Total Trades", metrics['total_trades'])
                st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
            with col3:
                st.metric("Avg R Multiple", f"{metrics['avg_r']:.2f}")
                st.metric("Max R Multiple", f"{metrics['max_r']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{abs(metrics['max_drawdown']):.2f}%")
            
            # Equity Curve
            st.subheader("Equity Curve")
            if not equity_curve.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(equity_curve['date'], equity_curve['balance'])
                ax.set_xlabel('Date')
                ax.set_ylabel('Account Balance ($)')
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("No equity curve data available.")
            
            # Health Gauge & Signals
            st.subheader("Health Gauge and Trade Signals")
            if not health_df.empty and not signals.empty:
                plot_data = pd.merge(health_df, signals[['date', 'buy_signal', 'sell_signal']], on='date', how='left').fillna(False)
                close_col = "close" if "close" in signals.columns else "Close"
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax1.plot(plot_data['date'], plot_data['health_gauge'], label='Health Gauge')
                ax1.axhline(buy_threshold, color='g', linestyle='--', label=f'Buy Threshold ({buy_threshold:.1f})')
                ax1.axhline(sell_threshold, color='r', linestyle='--', label=f'Sell Threshold ({sell_threshold:.1f})')
                ax1.scatter(plot_data.loc[plot_data['buy_signal'], 'date'], plot_data.loc[plot_data['buy_signal'], 'health_gauge'], marker='^', color='g', s=100, label='Buy Signal')
                ax1.scatter(plot_data.loc[plot_data['sell_signal'], 'date'], plot_data.loc[plot_data['sell_signal'], 'health_gauge'], marker='v', color='r', s=100, label='Sell Signal')
                ax1.set_ylabel('Health Gauge')
                ax1.legend()
                ax1.grid(True)
                
                merged_data = pd.merge(plot_data, signals[['date', close_col]], on='date', how='left')
                ax2.plot(merged_data['date'], merged_data[close_col], label='Price')
                ax2.scatter(merged_data.loc[merged_data['buy_signal'], 'date'], merged_data.loc[merged_data['buy_signal'], close_col], marker='^', color='g', s=100)
                ax2.scatter(merged_data.loc[merged_data['sell_signal'], 'date'], merged_data.loc[merged_data['sell_signal'], close_col], marker='v', color='r', s=100)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Price')
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No health gauge or signal data available.")
            
            # Trade Analysis
            st.subheader("Trade Analysis")
            if not trades_df.empty:
                trade_summary = trades_df[trades_df['action'].str.contains('entry|exit')].copy()
                trade_summary['date'] = pd.to_datetime(trade_summary['date']).dt.strftime('%Y-%m-%d')
                trade_summary['balance'] = trade_summary['balance'].map('${:,.2f}'.format)
                trade_summary['r_multiple'] = trade_summary['r_multiple'].map('{:+.2f}R'.format)
                st.dataframe(trade_summary[['date', 'trade_id', 'action', 'price', 'balance', 'r_multiple', 'drawdown']], height=400)
                
                r_values = trades_df[trades_df['action'].str.contains('exit')]['r_multiple'].dropna()
                if len(r_values) > 0:
                    st.subheader("R-Multiple Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(r_values, kde=True, ax=ax)
                    ax.axvline(0, color='black', linestyle='--')
                    ax.set_xlabel('R-Multiple')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            else:
                st.warning("No trades executed in this backtest.")

if __name__ == "__main__":

    # --- Set Ticker ---
    ticker = "GC=F"  # Example: Gold Futures

    # --- Fetch and Process Data ---
    price_df = fetch_price_data(ticker)
    latest_report, _ = get_last_two_reports(client)
    cot_df = latest_report  # Latest COT report
    df = merge_cot_price(cot_df, price_df)
    df = generate_features(df)
    df = generate_signals(df)
    df = calculate_health(df)
    df = calculate_bands(df)
    df = backtest(df)

    # --- Build Dashboard ---
    build_dashboard(df, ticker)
