import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import datetime
import time
import random
from scipy.stats import norm

# Set page configuration
st.set_page_config(
    page_title="Market Inefficiencies Monitor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to match the dashboard style
st.markdown("""
<style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --accent: #e74c3c;
        --light: #ecf0f1;
        --dark: #34495e;
        --success: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
    }
    
    .stApp {
        background-color: #f5f7fa;
    }
    
    h1, h2, h3 {
        color: var(--primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-container {
        background-color: var(--primary);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .positive {
        color: var(--success);
    }
    
    .negative {
        color: var(--danger);
    }
    
    .neutral {
        color: var(--warning);
    }
    
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .event {
        padding: 10px;
        margin-bottom: 10px;
        border-left: 4px solid var(--secondary);
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    .event.positive {
        border-left-color: var(--success);
        background-color: rgba(46, 204, 113, 0.1);
    }
    
    .event.negative {
        border-left-color: var(--danger);
        background-color: rgba(231, 76, 60, 0.1);
    }
    
    .signal-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .signal-tag.buy {
        background-color: rgba(46, 204, 113, 0.2);
        color: var(--success);
    }
    
    .signal-tag.sell {
        background-color: rgba(231, 76, 60, 0.2);
        color: var(--danger);
    }
    
    .signal-tag.neutral {
        background-color: rgba(243, 156, 18, 0.2);
        color: var(--warning);
    }
    
    /* Dashboard controls styling */
    .stSelectbox, .stMultiselect {
        background-color: white;
        border-radius: 4px;
    }
    
    .stButton > button {
        background-color: var(--secondary);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Custom divider */
    .divider {
        height: 3px;
        background-color: var(--light);
        margin: 1rem 0;
        border-radius: 2px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 5px;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--dark);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-container"><h1>Real-Time Market Inefficiencies Monitor</h1></div>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Monitor Settings")
    
    asset = st.selectbox(
        "Select Asset",
        ["Oil (WTI)", "Oil (Brent)", "Gold", "USD/EUR", "USD/CNY"]
    )
    
    regions = st.multiselect(
        "Region Focus",
        ["USA", "European Union", "Australia", "China"],
        default=["USA", "European Union"]
    )
    
    sentiment_method = st.selectbox(
        "Sentiment Analysis Method",
        ["FinBERT", "Lexicon-based", "Google Cloud NLP"]
    )
    
    refresh_rate = st.slider(
        "Refresh Rate (seconds)",
        min_value=5,
        max_value=60,
        value=30
    )
    
    alert_threshold = st.slider(
        "Sentiment Alert Threshold",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Microstructure Settings")
    
    volume_threshold = st.slider(
        "Volume Imbalance Threshold",
        min_value=1.5,
        max_value=5.0,
        value=2.5,
        step=0.1
    )
    
    volatility_window = st.slider(
        "Volatility Window (minutes)",
        min_value=5,
        max_value=60,
        value=15,
        step=5
    )
    
    orderflow_sensitivity = st.slider(
        "Order Flow Sensitivity",
        min_value=1,
        max_value=10,
        value=5
    )
    
    auto_trade = st.checkbox("Enable Auto-Trading Signals", value=False)
    
    if auto_trade:
        risk_per_trade = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        max_concurrent_trades = st.number_input("Max Concurrent Trades", min_value=1, max_value=10, value=3, step=1)

# Function to generate sample data for demo
def generate_sample_data(asset_name, timeframe="1D"):
    # Generate timestamps
    end_time = datetime.datetime.now()
    
    if timeframe == "1D":
        # 1-minute bars for a day
        periods = 390  # Typical trading day
        start_time = end_time - datetime.timedelta(hours=6, minutes=30)
        timestamps = [start_time + datetime.timedelta(minutes=i) for i in range(periods)]
    else:
        # 5-minute bars for a week
        periods = 5 * 24 * 7  # Week of 5-minute data
        start_time = end_time - datetime.timedelta(days=7)
        timestamps = [start_time + datetime.timedelta(minutes=i*5) for i in range(periods)]
    
    # Base price depends on asset
    if "Oil" in asset_name:
        base_price = 75.0
        tick_size = 0.01
    elif "Gold" in asset_name:
        base_price = 2000.0
        tick_size = 0.1
    else:  # FX pairs
        base_price = 1.1 if "EUR" in asset_name else 7.2  # USD/CNY
        tick_size = 0.0001
    
    # Generate price data with random walk + some patterns
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, 0.001, periods)
    
    # Add some trends and patterns
    trend = np.linspace(0, 0.01, periods)  # Slight uptrend
    volatility = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.002  # Oscillating volatility
    
    # Combine components
    cumulative_returns = np.cumsum(returns + trend + volatility)
    prices = base_price * (1 + cumulative_returns)
    
    # Generate OHLC data
    data = []
    for i, timestamp in enumerate(timestamps):
        price = prices[i]
        
        # Random high-low range based on volatility
        vol_factor = abs(volatility[i]) * 10
        high = price * (1 + 0.001 * (1 + vol_factor))
        low = price * (1 - 0.001 * (1 + vol_factor))
        
        # Random open within previous candle range
        if i == 0:
            open_price = price * 0.999
        else:
            prev_high = data[i-1]['high']
            prev_low = data[i-1]['low']
            open_price = data[i-1]['close']
        
        # Generate volume with some spikes
        base_volume = np.random.randint(100, 500)
        volume_spike = 1.0
        if i % 30 == 0:  # Volume spike every 30 bars
            volume_spike = np.random.randint(3, 10)
        
        volume = int(base_volume * volume_spike)
        
        # Add bid-ask microstructure
        spread = tick_size * np.random.randint(1, 5)
        bid_volume = int(volume * np.random.uniform(0.3, 0.7))
        ask_volume = volume - bid_volume
        
        # Create delta (imbalance)
        delta = bid_volume - ask_volume
        
        # Order flow metrics
        cumulative_delta = sum([d['delta'] for d in data[-10:]] + [delta]) if i > 0 else delta
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'delta': delta,
            'cumulative_delta': cumulative_delta,
            'spread': spread
        })
    
    return pd.DataFrame(data)

# Function to generate sample news events
def generate_sample_news():
    recent_events = [
        {
            "date": datetime.datetime.now() - datetime.timedelta(hours=2, minutes=15),
            "title": "OPEC+ Discusses Potential Production Increase",
            "content": "OPEC+ members are considering a production increase of 500,000 barrels per day at their next meeting, according to sources familiar with the discussions.",
            "sentiment": -0.68,
            "source": "Reuters",
            "region": "Global",
            "event_type": "Supply"
        },
        {
            "date": datetime.datetime.now() - datetime.timedelta(hours=6, minutes=30),
            "title": "Fed Official Signals Potential Rate Cut",
            "content": "A Federal Reserve official hinted at a possible interest rate cut in the next meeting, citing slowing inflation and concerns about economic growth.",
            "sentiment": 0.75,
            "source": "Bloomberg",
            "region": "USA",
            "event_type": "Economic"
        },
        {
            "date": datetime.datetime.now() - datetime.timedelta(days=1, hours=4),
            "title": "EU Approves New Energy Infrastructure Package",
            "content": "The European Union has approved a €200 billion energy infrastructure package focused on renewable energy development and grid modernization.",
            "sentiment": 0.62,
            "source": "Financial Times",
            "region": "European Union",
            "event_type": "Regulatory"
        },
        {
            "date": datetime.datetime.now() - datetime.timedelta(days=1, hours=7),
            "title": "China Reports Lower Than Expected Manufacturing Activity",
            "content": "China's manufacturing PMI came in at 49.2, below the expected 50.1, indicating contraction in the manufacturing sector for the third consecutive month.",
            "sentiment": -0.55,
            "source": "Caixin",
            "region": "China",
            "event_type": "Economic"
        }
    ]
    
    return recent_events

# Function to calculate microstructure metrics
def calculate_microstructure_metrics(df):
    # Sample window for calculations
    window = min(20, len(df))
    recent_data = df.tail(window)
    
    # 1. Order Flow Imbalance
    recent_delta = recent_data['delta'].sum()
    delta_zscore = (recent_delta - recent_data['delta'].mean()) / max(recent_data['delta'].std(), 1)
    
    # 2. Volume Profile Analysis
    volume_profile = recent_data.groupby(pd.cut(recent_data['close'], bins=10))['volume'].sum()
    max_volume_price = volume_profile.idxmax().mid
    current_price = df['close'].iloc[-1]
    dist_from_value_area = (current_price - max_volume_price) / current_price
    
    # 3. Volatility Analysis
    recent_returns = recent_data['close'].pct_change().dropna()
    realized_vol = recent_returns.std() * np.sqrt(252 * 390)  # Annualized from minute data
    
    # 4. Tick Analysis - Momentum
    momentum = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100  # 5-bar momentum
    
    # 5. Bid-Ask Spread Analysis
    avg_spread = recent_data['spread'].mean()
    normalized_spread = avg_spread / current_price * 10000  # Normalized in basis points
    
    # Compilation of metrics
    metrics = {
        'order_flow_imbalance': delta_zscore,
        'value_area_distance': dist_from_value_area,
        'realized_volatility': realized_vol,
        'momentum': momentum,
        'normalized_spread': normalized_spread
    }
    
    return metrics

# Function to detect entry signals based on combined sentiment and microstructure
def detect_entry_signals(sentiment_score, microstructure_metrics, thresholds):
    # Signal score starts at neutral
    signal_score = 0
    signal_type = "neutral"
    signal_strength = 0
    signal_reasons = []
    
    # 1. Evaluate sentiment extremes
    if abs(sentiment_score) > thresholds['sentiment']:
        # Strong sentiment signals mean-reversion opportunity
        signal_score = -1 * np.sign(sentiment_score)
        signal_reasons.append(f"Extreme {'positive' if sentiment_score > 0 else 'negative'} sentiment")
    
    # 2. Evaluate order flow for exhaustion
    if abs(microstructure_metrics['order_flow_imbalance']) > thresholds['order_flow']:
        # If order flow is showing exhaustion in the direction of sentiment
        if np.sign(microstructure_metrics['order_flow_imbalance']) == np.sign(sentiment_score):
            signal_score -= 0.5 * np.sign(sentiment_score)
            signal_reasons.append("Order flow exhaustion confirms sentiment extreme")
    
    # 3. Distance from value area
    if abs(microstructure_metrics['value_area_distance']) > thresholds['value_area']:
        # If price is extended from value area in the direction of sentiment
        if np.sign(microstructure_metrics['value_area_distance']) == np.sign(sentiment_score):
            signal_score -= 0.5 * np.sign(sentiment_score)
            signal_reasons.append("Price extended from value area")
    
    # 4. Volatility expansion can signal exhaustion
    if microstructure_metrics['realized_volatility'] > thresholds['volatility']:
        signal_score -= 0.3 * np.sign(sentiment_score)
        signal_reasons.append("Volatility expansion indicating potential reversal")
    
    # 5. Momentum confirmation
    if abs(microstructure_metrics['momentum']) > thresholds['momentum']:
        if np.sign(microstructure_metrics['momentum']) != np.sign(signal_score):
            signal_score *= 1.2  # Strengthen signal if momentum confirms
            signal_reasons.append("Momentum confirms potential reversal")
    
    # Determine final signal type and strength
    signal_strength = abs(signal_score)
    
    if signal_score > 0.7:
        signal_type = "buy"
    elif signal_score < -0.7:
        signal_type = "sell"
    else:
        signal_type = "neutral"
    
    return {
        'type': signal_type,
        'strength': signal_strength,
        'score': signal_score,
        'reasons': signal_reasons
    }

# Main dashboard layout
col1, col2, col3 = st.columns([1, 1, 1])

# Generate sample metrics
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h3>Current Sentiment Score</h3>", unsafe_allow_html=True)
    sentiment_value = random.uniform(-0.8, 0.8)
    sentiment_class = "positive" if sentiment_value > 0.1 else "negative" if sentiment_value < -0.1 else "neutral"
    st.markdown(f'<div class="metric-value {sentiment_class}">{sentiment_value:.2f}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h3>Microstructure Signal</h3>", unsafe_allow_html=True)
    micro_signal = random.choice(["Buy", "Sell", "Neutral"])
    micro_class = "buy" if micro_signal == "Buy" else "sell" if micro_signal == "Sell" else "neutral"
    st.markdown(f'<div class="signal-tag {micro_class}">{micro_signal}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<h3>Combined Signal Strength</h3>", unsafe_allow_html=True)
    signal_strength = random.uniform(0, 1)
    strength_class = "positive" if signal_strength > 0.7 else "neutral" if signal_strength > 0.4 else "negative"
    st.markdown(f'<div class="metric-value {strength_class}">{signal_strength:.2f}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Generate sample data
df = generate_sample_data(asset)
news_events = generate_sample_news()

# Calculate microstructure metrics
microstructure_metrics = calculate_microstructure_metrics(df)

# Set thresholds for signal detection
thresholds = {
    'sentiment': alert_threshold,
    'order_flow': orderflow_sensitivity / 10,
    'value_area': 0.005,
    'volatility': 0.5,
    'momentum': 1.0
}

# Detect signals
signal = detect_entry_signals(sentiment_value, microstructure_metrics, thresholds)

# Price chart with microstructure indicators
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.subheader(f"{asset} Price with Microstructure Indicators")

fig = make_subplots(rows=3, cols=1, 
                   shared_xaxes=True, 
                   vertical_spacing=0.03, 
                   row_heights=[0.6, 0.2, 0.2])

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="OHLC"
    ),
    row=1, col=1
)

# Add volume profile as histogram on the right
volume_profile = df.groupby(pd.cut(df['close'], bins=15))['volume'].sum()
fig.add_trace(
    go.Bar(
        x=volume_profile.values,
        y=[b.mid for b in volume_profile.index],
        orientation='h',
        marker=dict(color='rgba(58, 71, 80, 0.6)'),
        name="Volume Profile",
        showlegend=False,
        opacity=0.3,
        xaxis='x2',
    ),
    row=1, col=1
)

# Add volume bars
fig.add_trace(
    go.Bar(
        x=df['timestamp'],
        y=df['volume'],
        marker=dict(color='rgba(52, 152, 219, 0.5)'),
        name="Volume"
    ),
    row=2, col=1
)

# Add Order Flow Delta
fig.add_trace(
    go.Bar(
        x=df['timestamp'],
        y=df['delta'],
        marker=dict(
            color=np.where(df['delta'] >= 0, 'rgba(46, 204, 113, 0.7)', 'rgba(231, 76, 60, 0.7)')
        ),
        name="Order Flow Delta"
    ),
    row=3, col=1
)

# Add entry signal markers
if signal['type'] != "neutral":
    # Get the last 10 points where we might want to place markers
    last_points = df['timestamp'].tail(10)
    
    marker_color = 'rgba(46, 204, 113, 0.9)' if signal['type'] == 'buy' else 'rgba(231, 76, 60, 0.9)'
    
    # Place markers at strategic points
    fig.add_trace(
        go.Scatter(
            x=[last_points.iloc[-1]],
            y=[df['close'].iloc[-1] * (0.99 if signal['type'] == 'buy' else 1.01)],
            mode='markers',
            marker=dict(
                symbol='triangle-up' if signal['type'] == 'buy' else 'triangle-down',
                color=marker_color,
                size=15,
                line=dict(width=2, color='white')
            ),
            name=f"{signal['type'].upper()} Signal"
        ),
        row=1, col=1
    )

# Update layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis2=dict(
        overlaying='x',
        side='top',
        showticklabels=False
    )
)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="Delta", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)


