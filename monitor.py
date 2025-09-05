# ==============================
# Chunk 1/5
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import threading
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ==============================
# Page Setup
# ==============================
st.set_page_config(page_title="Market Monitor", layout="wide")

# ==============================
# Constants
# ==============================
API_KEY = st.secrets.get("FINAGE_API_KEY") or st.text_input("Enter Finage API Key", type="password")
REST_API_BASE_URL = "https://api.finage.co.uk"

MARKETS = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"],
    "Indices": ["SPX", "IXIC", "DJI", "FTSE", "DAX", "CAC", "N225", "HSI", "SSEC"],
    "Commodities": ["XAUUSD", "XAGUSD", "CL", "BZ", "NG"],
    "Metals": ["HG", "PL", "PA", "SI", "GC"]
}

TIMEFRAMES = {"1h": "hour", "4h": "hour", "1d": "day", "1w": "week", "1m": "month"}
CHART_TYPES = ["Candlestick", "Line", "Heikin-Ashi"]

# ==============================
# Data Cache
# ==============================
class DataCache:
    def __init__(self, cache_duration=3600):
        self.historical_data = {}
        self.live_data = {}
        self.last_update = {}
        self.cache_duration = cache_duration

    def set_historical(self, symbol, timeframe, data):
        key = f"{symbol}_{timeframe}"
        self.historical_data[key] = data
        self.last_update[key] = datetime.now()

    def get_historical(self, symbol, timeframe):
        key = f"{symbol}_{timeframe}"
        if key in self.historical_data:
            elapsed = (datetime.now() - self.last_update[key]).total_seconds()
            if elapsed < self.cache_duration:
                return self.historical_data[key]
        return None

    def update_live(self, symbol, data):
        self.live_data[symbol] = data

    def get_live(self, symbol):
        return self.live_data.get(symbol)

cache = DataCache()

# ==============================
# Rate Limiter
# ==============================
class RateLimiter:
    def __init__(self, calls_per_minute=5, calls_per_day=500):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.daily_calls = 0
        self.minute_calls = 0
        self.last_reset_minute = datetime.now()
        self.last_reset_day = datetime.now().date()
        self.lock = threading.Lock()

    def check_and_increment(self):
        with self.lock:
            now = datetime.now()
            today = now.date()
            if today > self.last_reset_day:
                self.daily_calls = 0
                self.last_reset_day = today
            if (now - self.last_reset_minute).total_seconds() >= 60:
                self.minute_calls = 0
                self.last_reset_minute = now
            if self.daily_calls >= self.calls_per_day or self.minute_calls >= self.calls_per_minute:
                return False
            self.daily_calls += 1
            self.minute_calls += 1
            return True

rate_limiter = RateLimiter()

# ==============================
# Chunk 2/5
# ==============================
# ==============================
# Live Price Fetch
# ==============================
def fetch_live_price(symbol):
    if not API_KEY:
        return None
    if not rate_limiter.check_and_increment():
        st.warning(f"Rate limit reached. Skipping live data for {symbol}")
        return None

    if any(symbol in MARKETS[cat] for cat in ["Forex", "Metals"]):
        endpoint = f"{REST_API_BASE_URL}/last/forex/{symbol}"
    elif any(symbol in MARKETS[cat] for cat in ["Indices"]):
        endpoint = f"{REST_API_BASE_URL}/last/index/{symbol}"
    else:
        endpoint = f"{REST_API_BASE_URL}/last/stock/{symbol}"

    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            cache.update_live(symbol, {
                "price": data.get("price", 0),
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
                "volume": data.get("volume", 0)
            })
            return data
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None

def update_live_prices(symbols):
    for symbol in symbols:
        fetch_live_price(symbol)

# ==============================
# Historical Data Fetch
# ==============================
def fetch_historical_data(symbol, timeframe="day", days=30):
    if not API_KEY:
        return None
    if not rate_limiter.check_and_increment():
        st.warning(f"Rate limit reached. Skipping historical data for {symbol}")
        return None

    cached = cache.get_historical(symbol, timeframe)
    if cached:
        return cached

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    from_date, to_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    if any(symbol in MARKETS[cat] for cat in ["Forex", "Metals"]):
        endpoint = f"{REST_API_BASE_URL}/agg/forex/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    elif any(symbol in MARKETS[cat] for cat in ["Indices"]):
        endpoint = f"{REST_API_BASE_URL}/agg/index/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    else:
        endpoint = f"{REST_API_BASE_URL}/agg/stock/{symbol}/1/{timeframe}/{from_date}/{to_date}"

    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            cache.set_historical(symbol, timeframe, data)
            return data
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

# ==============================
# Chart Data Preparation
# ==============================
def prepare_chart_data(symbol, timeframe="day", days=30):
    data = fetch_historical_data(symbol, timeframe, days)
    if not data or "results" not in data:
        st.error(f"No historical data available for {symbol}")
        return []

    results = data["results"]
    chart_data = []

    for item in results:
        chart_data.append({
            "time": item["t"] / 1000,  # milliseconds -> seconds
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"],
            "volume": item.get("v", 0)
        })
    return chart_data

# ==============================
# Heikin-Ashi Calculation
# ==============================
def calculate_heikin_ashi(ohlc_data):
    ha_data = []
    for i, candle in enumerate(ohlc_data):
        if i == 0:
            ha_open = (candle["open"] + candle["close"]) / 2
            ha_close = (candle["open"] + candle["high"] + candle["low"] + candle["close"]) / 4
            ha_high = candle["high"]
            ha_low = candle["low"]
        else:
            prev_ha = ha_data[-1]
            ha_open = (prev_ha["open"] + prev_ha["close"]) / 2
            ha_close = (candle["open"] + candle["high"] + candle["low"] + candle["close"]) / 4
            ha_high = max(candle["high"], ha_open, ha_close)
            ha_low = min(candle["low"], ha_open, ha_close)

        ha_data.append({
            "time": candle["time"],
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "volume": candle["volume"]
        })
    return ha_data

# ==============================
# Chunk 3/5
# ==============================
# ==============================
# Alert System
# ==============================
class AlertSystem:
    def __init__(self):
        self.alerts = {}

    def add_alert(self, symbol, condition, target_price, message):
        if symbol not in self.alerts:
            self.alerts[symbol] = []
        self.alerts[symbol].append({
            "condition": condition,  # "above" or "below"
            "price": target_price,
            "message": message,
            "triggered": False,
            "created_at": datetime.now()
        })

    def check_alerts(self, symbol, current_price):
        if symbol not in self.alerts:
            return []

        triggered_alerts = []
        for alert in self.alerts[symbol]:
            if alert["triggered"]:
                continue
            if alert["condition"] == "above" and current_price >= alert["price"]:
                alert["triggered"] = True
                triggered_alerts.append(alert["message"])
            elif alert["condition"] == "below" and current_price <= alert["price"]:
                alert["triggered"] = True
                triggered_alerts.append(alert["message"])

        # Remove triggered alerts
        self.alerts[symbol] = [a for a in self.alerts[symbol] if not a["triggered"]]
        return triggered_alerts

    def get_active_alerts(self, symbol=None):
        if symbol:
            return self.alerts.get(symbol, [])
        return self.alerts

# Initialize alert system
alert_system = AlertSystem()

# ==============================
# TradingView Chart Rendering
# ==============================
def render_tradingview_chart(symbol, data, container_id, chart_type="Candlestick", height=400):
    if chart_type.lower() == "heikin-ashi":
        data = calculate_heikin_ashi(data)
    
    # Convert data to JSON
    chart_json = json.dumps(data)

    # Basic TradingView Widget HTML
    html = f"""
    <div id="{container_id}"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
        new TradingView.widget({{
            "container_id": "{container_id}",
            "autosize": true,
            "symbol": "{symbol}",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "light",
            "style": 1,
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "details": true,
            "datafeed": {{
                onReady: cb => {{ cb({{}}); }},
                searchSymbols: (userInput, exchange, symbolType, onResultReadyCallback) => {{
                    onResultReadyCallback([]);
                }},
                resolveSymbol: (symbolName, onSymbolResolvedCallback, onResolveErrorCallback) => {{
                    onSymbolResolvedCallback({{
                        name: symbolName,
                        ticker: symbolName,
                        type: 'stock',
                        session: '24x7',
                        timezone: 'Etc/UTC',
                        exchange: '',
                        minmov: 1,
                        pricescale: 100,
                        has_intraday: true
                    }});
                }},
                getBars: (symbolInfo, resolution, from, to, onHistoryCallback, onErrorCallback, firstDataRequest) => {{
                    var bars = {chart_json};
                    onHistoryCallback(bars, {{noData: bars.length === 0}});
                }},
            }},
            "studies": [],
            "container_id": "{container_id}"
        }});
    </script>
    """
    return html

# ==============================
# Chunk 4/5
# ==============================
# ==============================
# Streamlit Dashboard
# ==============================
def build_dashboard():
    st.title("Market Monitor Dashboard")

    # Initialize session state
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now() - timedelta(minutes=10)
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = list(MARKETS.keys())[0]
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = MARKETS[st.session_state.selected_category][0]
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = "1d"
    if 'selected_chart_type' not in st.session_state:
        st.session_state.selected_chart_type = "Candlestick"

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Market Category
        selected_category = st.selectbox(
            "Market Category",
            list(MARKETS.keys()),
            index=list(MARKETS.keys()).index(st.session_state.selected_category)
        )
        if selected_category != st.session_state.selected_category:
            st.session_state.selected_category = selected_category
            st.session_state.selected_symbol = MARKETS[selected_category][0]

        # Symbol Selector
        available_symbols = MARKETS[selected_category]
        selected_symbol = st.selectbox(
            "Symbol",
            available_symbols,
            index=available_symbols.index(st.session_state.selected_symbol)
        )
        if selected_symbol != st.session_state.selected_symbol:
            st.session_state.selected_symbol = selected_symbol

        # Chart Settings
        st.subheader("Chart Settings")
        selected_timeframe = st.selectbox(
            "Timeframe",
            list(TIMEFRAMES.keys()),
            index=list(TIMEFRAMES.keys()).index(st.session_state.selected_timeframe)
        )
        if selected_timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = selected_timeframe

        selected_chart_type = st.selectbox(
            "Chart Type",
            CHART_TYPES,
            index=CHART_TYPES.index(st.session_state.selected_chart_type)
        )
        if selected_chart_type != st.session_state.selected_chart_type:
            st.session_state.selected_chart_type = selected_chart_type

        # History range and refresh
        days_lookback = st.slider("Days of History", 7, 90, 30)
        use_real_api = st.checkbox("Use Real API (if API Key provided)", value=True)
        refresh_minutes = st.number_input("Update Frequency (minutes)", 1, 360, 60)
        refresh_seconds = refresh_minutes * 60

        # Manual refresh
        if st.button("Refresh Data Now"):
            update_live_prices([selected_symbol])
            st.session_state.last_update_time = datetime.now()
            st.success(f"Data updated at {st.session_state.last_update_time.strftime('%H:%M:%S')}")

        # Alert creation
        current_data = cache.get_live(selected_symbol)
        current_price = current_data["price"] if current_data else 100

        alert_price = st.number_input("Alert Price", value=current_price, step=0.01)
        alert_condition = st.radio("Condition", ["above", "below"])
        alert_message = st.text_input("Alert Message", value=f"{selected_symbol} {alert_condition} {alert_price}")

        if st.button("Add Alert"):
            alert_system.add_alert(selected_symbol, alert_condition, alert_price, alert_message)
            st.success(f"Alert added for {selected_symbol}")

    # Auto-refresh using Streamlit
    st_autorefresh(interval=refresh_seconds * 1000, key="data_refresh")

    # Update live prices if refresh needed
    time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
    if time_since_update > refresh_seconds:
        update_live_prices([selected_symbol])
        st.session_state.last_update_time = datetime.now()

# ==============================
# Chunk 5/5
# ==============================
# ==============================
# Main Chart and Overview
# ==============================
    # Main chart area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header(f"{selected_symbol} Chart")
        timeframe_api = TIMEFRAMES[selected_timeframe]
        adjusted_days = min(days_lookback, 7 if selected_timeframe == "1h" else 30 if selected_timeframe == "4h" else days_lookback)
        chart_data = prepare_chart_data(selected_symbol, timeframe_api, adjusted_days)

        if selected_chart_type == "Heikin-Ashi":
            chart_data = calculate_heikin_ashi(chart_data)

        chart_container_id = f"chart_{selected_symbol}_{timeframe_api}_{selected_chart_type}"
        chart_html = render_tradingview_chart(
            selected_symbol,
            chart_data,
            chart_container_id,
            chart_type=selected_chart_type,
            height=500
        )
        st.components.v1.html(chart_html, height=520)

    with col2:
        st.header("Market Overview")
        live_data = {}
        for symbol in MARKETS[selected_category]:
            symbol_data = cache.get_live(symbol)
            if symbol_data:
                price = symbol_data["price"]
                timestamp = symbol_data["timestamp"]
            else:
                price = 0
                timestamp = int(time.time() * 1000)

            change = 0  # Real change requires previous close
            live_data[symbol] = {"price": price, "change": change, "timestamp": timestamp}

            if symbol_data:
                triggered_alerts = alert_system.check_alerts(symbol, price)
                for alert in triggered_alerts:
                    st.warning(f"ALERT: {alert}")

        df = pd.DataFrame([
            {"Symbol": s, "Price": round(d["price"], 4), "Change %": round(d["change"], 2),
             "Updated": datetime.fromtimestamp(d["timestamp"] / 1000).strftime('%H:%M:%S')}
            for s, d in live_data.items()
        ])
        st.dataframe(df, use_container_width=True)

# ==============================
# Main App Entry
# ==============================
if __name__ == "__main__":
    build_dashboard()


