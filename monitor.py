import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ==============================
# Page Setup
# ==============================
st.set_page_config(page_title="Market Monitor", layout="wide")

# ==============================
# Constants and Config
# ==============================
API_KEY = st.secrets.get("FINAGE_API_KEY") or st.text_input("Enter Finage API Key", type="password")
REST_API_BASE_URL = "https://api.finage.co.uk"

MARKETS = {
    "Forex": ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURJPY","GBPJPY","EURGBP"],
    "Indices": ["SPX","IXIC","DJI","FTSE","DAX","CAC","N225","HSI","SSEC"],
    "Commodities": ["XAUUSD","XAGUSD","CL","BZ","NG"],
    "Metals": ["HG","PL","PA","SI","GC"]
}

TIMEFRAMES = {"1h":"hour","4h":"hour","1d":"day","1w":"week","1m":"month"}
CHART_TYPES = ["Candlestick","Line","Heikin-Ashi"]

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
            if (datetime.now() - self.last_update[key]).total_seconds() < self.cache_duration:
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

    def check_and_increment(self):
        now = datetime.now()
        if now.date() > self.last_reset_day:
            self.daily_calls = 0
            self.last_reset_day = now.date()
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
# Alerts
# ==============================
class AlertSystem:
    def __init__(self):
        self.alerts = {}

    def add_alert(self, symbol, condition, price, message):
        self.alerts.setdefault(symbol, []).append({"condition": condition, "price": price, "message": message, "triggered": False})

    def check_alerts(self, symbol, current_price):
        triggered = []
        for alert in self.alerts.get(symbol, []):
            if not alert["triggered"]:
                if (alert["condition"]=="above" and current_price>=alert["price"]) or (alert["condition"]=="below" and current_price<=alert["price"]):
                    alert["triggered"] = True
                    triggered.append(alert["message"])
        self.alerts[symbol] = [a for a in self.alerts.get(symbol,[]) if not a["triggered"]]
        return triggered

alert_system = AlertSystem()

# ==============================
# Auto Refresh
# ==============================
st_autorefresh(interval=60000, key="refresh")  # Refresh every 60s

# ==============================
# Live Price Fetch
# ==============================
def fetch_live_price(symbol):
    if not API_KEY or not rate_limiter.check_and_increment():
        return None

    if symbol in MARKETS["Forex"] + MARKETS["Metals"]:
        endpoint = f"{REST_API_BASE_URL}/last/forex/{symbol}"
    elif symbol in MARKETS["Indices"]:
        endpoint = f"{REST_API_BASE_URL}/last/index/{symbol}"
    else:
        endpoint = f"{REST_API_BASE_URL}/last/stock/{symbol}"

    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            cache.update_live(symbol, {
                "price": data.get("price", 0),
                "timestamp": data.get("timestamp", int(time.time()*1000)),
                "volume": data.get("volume", 0)
            })
            return data
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
    if not API_KEY or not rate_limiter.check_and_increment():
        return None

    cached = cache.get_historical(symbol, timeframe)
    if cached:
        return cached

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")

    if symbol in MARKETS["Forex"] + MARKETS["Metals"]:
        endpoint = f"{REST_API_BASE_URL}/agg/forex/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    elif symbol in MARKETS["Indices"]:
        endpoint = f"{REST_API_BASE_URL}/agg/index/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    else:
        endpoint = f"{REST_API_BASE_URL}/agg/stock/{symbol}/1/{timeframe}/{from_date}/{to_date}"

    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            cache.set_historical(symbol, timeframe, data)
            return data
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
        return []

    chart_data = []
    for item in data["results"]:
        chart_data.append({
            "time": item["t"]/1000,
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
        if i==0:
            ha_open = (candle["open"] + candle["close"])/2
            ha_close = (candle["open"] + candle["high"] + candle["low"] + candle["close"])/4
        else:
            prev_ha = ha_data[-1]
            ha_open = (prev_ha["open"] + prev_ha["close"])/2
            ha_close = (candle["open"] + candle["high"] + candle["low"] + candle["close"])/4
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
# TradingView Chart Rendering
# ==============================
def render_tradingview_chart(symbol, chart_data, container_id, chart_type="Candlestick", height=500):
    if chart_type == "Heikin-Ashi":
        chart_data = calculate_heikin_ashi(chart_data)

    # Prepare JS data array
    js_data = [{"time": int(d["time"]), "open": d["open"], "high": d["high"],
                "low": d["low"], "close": d["close"], "volume": d["volume"]} for d in chart_data]

    html_template = f"""
    <div id="{container_id}"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    const data = {json.dumps(js_data)};
    new TradingView.widget({{
        "container_id": "{container_id}",
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": {0 if chart_type=="Candlestick" else 1},
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "details": true,
        "studies": [],
        "datafeed": {{
            onReady: cb => cb({{supported_resolutions: ["1","5","15","60","240","D","W","M"]}}),
            searchSymbols: (userInput, exchange, symbolType, onResultReadyCallback) => {},
            resolveSymbol: (symbolName, onSymbolResolvedCallback, onResolveErrorCallback) => {{
                onSymbolResolvedCallback({{
                    name: symbolName,
                    ticker: symbolName,
                    type: "stock",
                    session: "24x7",
                    timezone: "Etc/UTC",
                    minmov: 1,
                    pricescale: 100,
                    has_intraday: true,
                    intraday_multipliers: ["1","5","15","60","240"],
                    supported_resolution: ["1","5","15","60","240","D","W","M"]
                }});
            }},
            getBars: (symbolInfo, resolution, periodParams, onHistoryCallback, onErrorCallback) => {{
                onHistoryCallback(data, {{}}); 
            }},
            subscribeBars: () => {{}},
            unsubscribeBars: () => {{}}
        }},
    }});
    </script>
    """
    return html_template

# ==============================
# Sidebar & Asset Selection
# ==============================
def sidebar_controls():
    st.sidebar.header("Market Settings")

    selected_category = st.sidebar.selectbox("Market Category", list(MARKETS.keys()))
    symbols = MARKETS[selected_category]
    selected_symbol = st.sidebar.selectbox("Symbol", symbols)

    selected_timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
    selected_chart_type = st.sidebar.selectbox("Chart Type", CHART_TYPES)

    days_lookback = st.sidebar.slider("History (days)", 7, 90, 30)
    refresh_minutes = st.sidebar.number_input("Update Frequency (min)", 1, 360, 60)
    return selected_category, selected_symbol, selected_timeframe, selected_chart_type, days_lookback, refresh_minutes

    # ==============================
# Main Dashboard Rendering
# ==============================
def build_dashboard():
    st.title("Market Monitor Dashboard")

    # Sidebar selections
    category, symbol, timeframe, chart_type, days_lookback, refresh_minutes = sidebar_controls()
    refresh_seconds = refresh_minutes * 60

    # Initialize session state
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now() - timedelta(minutes=10)

    # Auto-refresh
    st_autorefresh(interval=refresh_seconds * 1000, key="data_refresh")

    # Update live prices if needed
    time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
    if time_since_update > refresh_seconds:
        update_live_prices([symbol])
        st.session_state.last_update_time = datetime.now()

    # Chart rendering
    st.header(f"{symbol} Chart")
    timeframe_api = TIMEFRAMES[timeframe]
    adjusted_days = min(days_lookback, 7 if timeframe == "1h" else 30 if timeframe == "4h" else days_lookback)
    chart_data = prepare_chart_data(symbol, timeframe_api, adjusted_days)

    chart_html = render_tradingview_chart(symbol, chart_data, f"chart_{symbol}", chart_type=chart_type, height=500)
    st.components.v1.html(chart_html, height=520)

    # Live Market Overview
    st.header("Market Overview")
    live_data = {}
    for s in MARKETS[category]:
        symbol_data = cache.get_live(s)
        price = symbol_data["price"] if symbol_data else 0
        timestamp = symbol_data["timestamp"] if symbol_data else int(time.time() * 1000)
        change = 0  # Can be enhanced with previous close
        live_data[s] = {"price": price, "change": change, "timestamp": timestamp}

        # Alert check
        if symbol_data:
            triggered_alerts = alert_system.check_alerts(s, price)
            for alert in triggered_alerts:
                st.warning(f"ALERT: {alert}")

    df = pd.DataFrame([
        {"Symbol": s, "Price": round(d["price"], 4), "Change %": round(d["change"], 2),
         "Updated": datetime.fromtimestamp(d["timestamp"]/1000).strftime('%H:%M:%S')}
        for s, d in live_data.items()
    ])
    st.dataframe(df, use_container_width=True)

# ==============================
# Main App Entry
# ==============================
if __name__ == "__main__":
    build_dashboard()
