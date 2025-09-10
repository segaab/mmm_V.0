import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahooquery import Ticker
import datetime
import time

# Page config
st.set_page_config(layout="wide", page_title="RVol Gap-Up Backtester")

# Dashboard title
st.title("RVol Gap-Up Strategy Backtester")
st.markdown("""
This dashboard backtests the Relative Volume (RVol) Gap-Up trading strategy using hourly data.
The strategy identifies volume spikes at session opens and trades based on the subsequent candle pattern.
""")

# ---------------- Sidebar ----------------
st.sidebar.header("Strategy Parameters")

# Ticker selection - batch asset testing
TICKER_MAP = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC-USD",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH-USD",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.": "DX-Y.NYB",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6N=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "^DJI",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE": "NQ=F",
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": "^N225",
    "SPDR S&P 500 ETF TRUST": "SPY",
    "COPPER - COMMODITY EXCHANGE INC.":"HG=F"
}

selected_assets = st.sidebar.multiselect(
    "Select Assets", options=list(TICKER_MAP.keys()),
    default=list(TICKER_MAP.keys())[:5],
    key="select_assets_rvol"
)

# Date range selection
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", pd.to_datetime("2023-01-01"), key="start_date_rvol")
end_date   = col2.date_input("End Date", pd.to_datetime("today"), key="end_date_rvol")

# RVol parameters
rvol_window   = st.sidebar.slider("RVol Window (hours)", 1, 24, 5, key="rvol_window_rvol")
gap_threshold = st.sidebar.slider("Gap-Up Threshold", 1.0, 5.0, 2.0, 0.1, key="gap_threshold_rvol")

# Risk management parameters
atr_window     = st.sidebar.slider("ATR Window (hours)", 5, 48, 14, key="atr_window_rvol")
sl_multiplier  = st.sidebar.slider("Stop Loss (x ATR)", 0.5, 3.0, 1.5, 0.1, key="sl_multiplier_rvol")
tp_multiplier  = st.sidebar.slider("Take Profit (x ATR)", 0.5, 5.0, 2.0, 0.1, key="tp_multiplier_rvol")

# Session selection
sessions = st.sidebar.multiselect(
    "Sessions to Test", ["Asian", "London", "New York"],
    ["Asian", "London", "New York"], key="sessions_rvol"
)

# Execute button
run_backtest = st.sidebar.button("Run Backtest", key="run_btn_rvol")

# Placeholder for messages
data_placeholder = st.empty()

# ---------------- Data Fetching ----------------
@st.cache_data(ttl=3600)
def get_data(ticker_symbol, start_date, end_date):
    data_placeholder.info(f"Fetching data for {ticker_symbol}...")
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str   = end_date.strftime('%Y-%m-%d')
    
    try:
        ticker = Ticker(ticker_symbol)
        df = ticker.history(interval="1h", start=start_str, end=end_str)
        
        # Reset multiindex if needed
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df = df.set_index('date')
        
        df.columns = [col.lower() for col in df.columns]
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in data")
        
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ---------------- Data Processing ----------------
def process_data(df, rvol_window, gap_threshold, atr_window):
    df = df.copy()
    
    # Rolling average volume & RVol
    df['volume_ma'] = df['volume'].rolling(window=rvol_window).mean()
    df['rvol'] = df['volume'] / df['volume_ma']
    
    # Hour extraction for session mapping
    df['hour_gmt'] = df.index.hour
    df['asian_open'] = df['hour_gmt'].between(3, 4)
    df['london_open'] = df['hour_gmt'].between(10, 11)
    df['ny_open'] = df['hour_gmt'].between(16, 17)
    
    # ATR calculation
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=atr_window).mean()
    
    # Candle features
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_to_range'] = df['body'] / (df['high'] - df['low'])
    
    # Candle classification
    df['candle_type'] = 'indecision'
    bullish_mask = (df['close'] > df['open']) & (df['body_to_range'] > 0.5)
    bearish_mask = (df['close'] < df['open']) & (df['body_to_range'] > 0.5)
    df.loc[bullish_mask, 'candle_type'] = 'bullish'
    df.loc[bearish_mask, 'candle_type'] = 'bearish'
    
    # Initialize gap columns
    for session in ['asian', 'london', 'ny']:
        df[f'{session}_gap'] = 0.0
        df[f'{session}_gapup'] = False
    
    # Calculate session gaps
    for session in ['asian', 'london', 'ny']:
        session_mask = df[f'{session}_open']
        session_times = df[session_mask].index
        for i in range(1, len(session_times)):
            curr_idx = session_times[i]
            prev_idx = session_times[i-1]
            if pd.notna(df.loc[curr_idx, 'rvol']) and pd.notna(df.loc[prev_idx, 'rvol']):
                df.loc[curr_idx, f'{session}_gap'] = df.loc[curr_idx, 'rvol'] / df.loc[prev_idx, 'rvol']
                df.loc[curr_idx, f'{session}_gapup'] = df.loc[curr_idx, f'{session}_gap'] >= gap_threshold
    
    return df

# ---------------- Signal Generation ----------------
def generate_signals(df, sessions, sl_multiplier, tp_multiplier):
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['exit_price'] = np.nan
    df['exit_time'] = pd.NaT
    df['pnl'] = np.nan
    df['trade_session'] = ''
    df['trade_id'] = 0
    
    active_trade = False
    entry_idx = None
    trade_direction = 0
    trade_id = 0
    
    for i in range(1, len(df)):
        curr_idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # Manage active trades
        if active_trade:
            curr_price = df.loc[curr_idx, 'close']
            if trade_direction > 0:  # Long
                if df.loc[curr_idx, 'low'] <= df.loc[entry_idx, 'stop_loss']:
                    df.loc[curr_idx, 'exit_price'] = df.loc[entry_idx, 'stop_loss']
                    df.loc[curr_idx, 'pnl'] = (df.loc[curr_idx, 'exit_price'] - df.loc[entry_idx, 'entry_price']) / df.loc[entry_idx, 'entry_price']
                    df.loc[curr_idx, 'exit_time'] = curr_idx
                    active_trade = False
                elif df.loc[curr_idx, 'high'] >= df.loc[entry_idx, 'take_profit']:
                    df.loc[curr_idx, 'exit_price'] = df.loc[entry_idx, 'take_profit']
                    df.loc[curr_idx, 'pnl'] = (df.loc[curr_idx, 'exit_price'] - df.loc[entry_idx, 'entry_price']) / df.loc[entry_idx, 'entry_price']
                    df.loc[curr_idx, 'exit_time'] = curr_idx
                    active_trade = False
            else:  # Short
                if df.loc[curr_idx, 'high'] >= df.loc[entry_idx, 'stop_loss']:
                    df.loc[curr_idx, 'exit_price'] = df.loc[entry_idx, 'stop_loss']
                    df.loc[curr_idx, 'pnl'] = (df.loc[entry_idx, 'entry_price'] - df.loc[curr_idx, 'exit_price']) / df.loc[entry_idx, 'entry_price']
                    df.loc[curr_idx, 'exit_time'] = curr_idx
                    active_trade = False
                elif df.loc[curr_idx, 'low'] <= df.loc[entry_idx, 'take_profit']:
                    df.loc[curr_idx, 'exit_price'] = df.loc[entry_idx, 'take_profit']
                    df.loc[curr_idx, 'pnl'] = (df.loc[entry_idx, 'entry_price'] - df.loc[curr_idx, 'exit_price']) / df.loc[entry_idx, 'entry_price']
                    df.loc[curr_idx, 'exit_time'] = curr_idx
                    active_trade = False
        
        if active_trade:
            continue
        
        # Generate new signals
        for session in sessions:
            session_lower = session.lower()
            if prev_idx in df.index and df.loc[prev_idx, f'{session_lower}_gapup']:
                candle_type = df.loc[curr_idx, 'candle_type']
                if candle_type == 'bullish':
                    df.loc[curr_idx, 'signal'] = 1
                    df.loc[curr_idx, 'entry_price'] = df.loc[curr_idx, 'close']
                    df.loc[curr_idx, 'stop_loss'] = df.loc[curr_idx, 'entry_price'] - (sl_multiplier * df.loc[curr_idx, 'atr'])
                    df.loc[curr_idx, 'take_profit'] = df.loc[curr_idx, 'entry_price'] + (tp_multiplier * df.loc[curr_idx, 'atr'])
                    df.loc[curr_idx, 'trade_session'] = session
                    trade_id += 1
                    df.loc[curr_idx, 'trade_id'] = trade_id
                    active_trade = True
                    entry_idx = curr_idx
                    trade_direction = 1
                elif candle_type == 'bearish':
                    df.loc[curr_idx, 'signal'] = -1
                    df.loc[curr_idx, 'entry_price'] = df.loc[curr_idx, 'close']
                    df.loc[curr_idx, 'stop_loss'] = df.loc[curr_idx, 'entry_price'] + (sl_multiplier * df.loc[curr_idx, 'atr'])
                    df.loc[curr_idx, 'take_profit'] = df.loc[curr_idx, 'entry_price'] - (tp_multiplier * df.loc[curr_idx, 'atr'])
                    df.loc[curr_idx, 'trade_session'] = session
                    trade_id += 1
                    df.loc[curr_idx, 'trade_id'] = trade_id
                    active_trade = True
                    entry_idx = curr_idx
                    trade_direction = -1
    
    # Close open trades at end
    if active_trade and entry_idx is not None:
        last_idx = df.index[-1]
        df.loc[last_idx, 'exit_price'] = df.loc[last_idx, 'close']
        if trade_direction > 0:
            df.loc[last_idx, 'pnl'] = (df.loc[last_idx, 'exit_price'] - df.loc[entry_idx, 'entry_price']) / df.loc[entry_idx, 'entry_price']
        else:
            df.loc[last_idx, 'pnl'] = (df.loc[entry_idx, 'entry_price'] - df.loc[last_idx, 'exit_price']) / df.loc[entry_idx, 'entry_price']
        df.loc[last_idx, 'exit_time'] = last_idx
    
    return df



# ---------------- Backtest Statistics ----------------
def calculate_stats(df):
    trades = df[df['signal'] != 0].copy()
    if trades.empty:
        return {
            'total_trades': 0, 'win_rate': 0, 'avg_win': 0,
            'avg_loss': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'max_drawdown': 0, 'session_stats': {}, 'trades': trades
        }

    for i, row in trades.iterrows():
        trade_id = row['trade_id']
        exit_rows = df[(df['exit_time'] > i) & (df['trade_id'] == trade_id)]
        if not exit_rows.empty:
            exit_row = exit_rows.iloc[0]
            trades.loc[i, 'exit_price'] = exit_row['exit_price']
            trades.loc[i, 'pnl'] = exit_row['pnl']
    
    trades.dropna(subset=['pnl'], inplace=True)
    if trades.empty:
        return {
            'total_trades': 0, 'win_rate': 0, 'avg_win': 0,
            'avg_loss': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'max_drawdown': 0, 'session_stats': {}, 'trades': trades
        }

    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    win_rate = len(winning_trades) / len(trades)
    avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    profit_factor = winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if not losing_trades.empty else float('inf')
    
    # Sharpe ratio (annualized)
    returns_mean = trades['pnl'].mean()
    returns_std = trades['pnl'].std()
    sharpe_ratio = (returns_mean / returns_std) * np.sqrt(252) if returns_std != 0 else 0

    # Max drawdown
    cumulative = (1 + trades['pnl']).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min() if not cumulative.empty else 0

    # Session stats
    session_stats = {}
    for session in trades['trade_session'].unique():
        session_trades = trades[trades['trade_session'] == session]
        session_wins = session_trades[session_trades['pnl'] > 0]
        session_stats[session] = {
            'total_trades': len(session_trades),
            'win_rate': len(session_wins)/len(session_trades),
            'avg_return': session_trades['pnl'].mean()
        }
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'session_stats': session_stats,
        'trades': trades
    }

# ---------------- Plotting ----------------
def plot_results(df, stats, gap_threshold):
    if 'trades' not in stats or stats['trades'].empty:
        st.warning("No trades found.")
        return None
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Price", "RVol", "Cumulative Returns"),
        row_heights=[0.5,0.2,0.3]
    )
    
    # Price candlesticks
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 name="Price"), row=1, col=1)
    
    # RVol
    fig.add_trace(go.Scatter(x=df.index, y=df['rvol'], mode='lines', name='RVol', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[gap_threshold]*len(df), mode='lines', name=f'Threshold ({gap_threshold})', line=dict(color='red', dash='dash')), row=2, col=1)
    
    # Signals
    buy_signals = df[df['signal']==1]
    sell_signals = df[df['signal']==-1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['entry_price'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=2, color='darkgreen'))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['entry_price'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=2, color='darkred'))), row=1, col=1)
    
    # Cumulative returns
    cumulative_returns = pd.Series(index=df.index)
    for idx, trade in stats['trades'].iterrows():
        cumulative_returns[idx] = trade['pnl']
    cumulative_returns.fillna(0, inplace=True)
    cumulative_returns = (1 + cumulative_returns).cumprod() - 1
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Cumulative Returns', line=dict(color='blue')), row=3, col=1)
    
    fig.update_layout(height=800, title="RVol Gap-Up Backtest", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

# ---------------- Streamlit Layout ----------------
st.header("Backtest Results")

# Run batch backtesting for all assets
import threading

results_dict = {}

def backtest_asset(name, symbol):
    df = get_data(symbol, start_date, end_date)
    if df is None or df.empty:
        return
    df_proc = process_data(df, rvol_window, gap_threshold, atr_window)
    df_signals = generate_signals(df_proc, [s.lower() for s in sessions], sl_multiplier, tp_multiplier)
    stats = calculate_stats(df_signals)
    results_dict[name] = (df_signals, stats)

threads = []
for batch in range(0, len(TICKER_MAP), 5):
    for name, symbol in list(TICKER_MAP.items())[batch:batch+5]:
        t = threading.Thread(target=backtest_asset, args=(name, symbol))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

# Display results per asset
for name, (df_signals, stats) in results_dict.items():
    st.subheader(f"{name} Backtest")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", stats['total_trades'])
    col2.metric("Win Rate", f"{stats['win_rate']:.2%}")
    col3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    col4.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    
    fig = plot_results(df_signals, stats, gap_threshold)
    if fig:
        st.plotly_chart(fig, use_container_width=True)