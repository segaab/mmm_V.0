import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
import threading
import pytz
import io
import base64
from streamlit_javascript import st_javascript
import random

st.set_page_config(page_title="Market Monitor", layout="wide")

# Constants
API_KEY = st.secrets["FINAGE_API_KEY"] if "FINAGE_API_KEY" in st.secrets else st.text_input("Enter Finage API Key", type="password")
REST_API_BASE_URL = "https://api.finage.co.uk"

# Market categories and symbols
MARKETS = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP"],
    "Indices": ["SPX", "IXIC", "DJI", "FTSE", "DAX", "CAC", "N225", "HSI", "SSEC"],
    "Commodities": ["XAUUSD", "XAGUSD", "CL", "BZ", "NG"],
    "Metals": ["HG", "PL", "PA", "SI", "GC"]
}

# Cache management
class DataCache:
    def __init__(self, cache_duration=3600):  # Cache duration in seconds (1 hour default)
        self.historical_data = {}
        self.live_data = {}
        self.last_update = {}
        self.cache_duration = cache_duration
        
    def set_historical(self, symbol, timeframe, data):
        cache_key = f"{symbol}_{timeframe}"
        self.historical_data[cache_key] = data
        self.last_update[cache_key] = datetime.now()
        
    def get_historical(self, symbol, timeframe):
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.historical_data:
            elapsed = (datetime.now() - self.last_update[cache_key]).total_seconds()
            if elapsed < self.cache_duration:
                return self.historical_data[cache_key]
        return None
    
    def update_live(self, symbol, data):
        self.live_data[symbol] = data
        
    def get_live(self, symbol):
        return self.live_data.get(symbol)
    
    def get_all_live(self):
        return self.live_data

# Initialize cache
cache = DataCache()

# Rate limiter for REST API
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
            current_time = datetime.now()
            current_date = current_time.date()
            
            # Reset daily counter if day changed
            if current_date > self.last_reset_day:
                self.daily_calls = 0
                self.last_reset_day = current_date
                
            # Reset minute counter if minute changed
            if (current_time - self.last_reset_minute).total_seconds() >= 60:
                self.minute_calls = 0
                self.last_reset_minute = current_time
                
            # Check if we can make a call
            if self.daily_calls >= self.calls_per_day:
                return False
            
            if self.minute_calls >= self.calls_per_minute:
                return False
                
            # Increment counters
            self.daily_calls += 1
            self.minute_calls += 1
            return True

# Initialize rate limiter
rate_limiter = RateLimiter()

# Function to fetch live price data via REST API
def fetch_live_price(symbol):
    if not API_KEY:
        return None
    
    # Check rate limiter
    if not rate_limiter.check_and_increment():
        st.warning(f"Rate limit reached. Skipping live data fetch for {symbol}")
        return None
    
    # Determine API endpoint based on symbol type
    if any(symbol in MARKETS[category] for category in ["Forex", "Metals"]):
        endpoint = f"{REST_API_BASE_URL}/last/forex/{symbol}"
    elif any(symbol in MARKETS[category] for category in ["Indices"]):
        endpoint = f"{REST_API_BASE_URL}/last/index/{symbol}"
    else:  # Default to stock/commodity endpoint
        endpoint = f"{REST_API_BASE_URL}/last/stock/{symbol}"
    
    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            # Update cache with live data
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

# Function to update live prices for multiple symbols
def update_live_prices(symbols, use_real_api=True):
    for symbol in symbols:
        if use_real_api and API_KEY:
            fetch_live_price(symbol)
        else:
            # Generate placeholder data
            base_price = 100
            if symbol.startswith("EUR"):
                base_price = 1.1
            elif symbol.startswith("GBP"):
                base_price = 1.3
            elif symbol.startswith("USD"):
                base_price = 0.9
            elif symbol.startswith("XAU"):
                base_price = 2000
            elif symbol.startswith("XAG"):
                base_price = 25
            
            # Get existing price if available or use base price
            current_data = cache.get_live(symbol)
            if current_data:
                current_price = current_data["price"]
            else:
                current_price = base_price
            
            # Add some random movement
            new_price = current_price * (1 + (random.random() - 0.5) * 0.005)  # ±0.25% change
            
            # Update cache
            cache.update_live(symbol, {
                "price": new_price,
                "timestamp": int(time.time() * 1000),
                "volume": int(random.random() * 10000)
            })

# Function to fetch historical data via REST API
def fetch_historical_data(symbol, timeframe="day", days=30):
    if not API_KEY:
        return None
    
    # Check rate limiter
    if not rate_limiter.check_and_increment():
        st.warning(f"Rate limit reached. Skipping historical data fetch for {symbol}")
        return None
    
    # Check cache first
    cached_data = cache.get_historical(symbol, timeframe)
    if cached_data is not None:
        return cached_data
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    # Determine API endpoint based on symbol type
    if any(symbol in MARKETS[category] for category in ["Forex", "Metals"]):
        endpoint = f"{REST_API_BASE_URL}/agg/forex/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    elif any(symbol in MARKETS[category] for category in ["Indices"]):
        endpoint = f"{REST_API_BASE_URL}/agg/index/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    else:  # Default to stock/commodity endpoint
        endpoint = f"{REST_API_BASE_URL}/agg/stock/{symbol}/1/{timeframe}/{from_date}/{to_date}"
    
    try:
        response = requests.get(f"{endpoint}?apikey={API_KEY}")
        if response.status_code == 200:
            data = response.json()
            # Cache the result
            cache.set_historical(symbol, timeframe, data)
            return data
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        return None

# Function to convert historical data to chart format
def prepare_chart_data(symbol, timeframe="day", days=30):
    data = fetch_historical_data(symbol, timeframe, days)
    if not data or "results" not in data:
        # Generate placeholder data if we don't have real data
        # This is useful for demo/testing without API key
        return generate_placeholder_data(symbol, timeframe, days)
    
    results = data["results"]
    chart_data = []
    
    for item in results:
        chart_data.append({
            "time": item["t"] / 1000,  # Convert milliseconds to seconds for chart
            "open": item["o"],
            "high": item["h"],
            "low": item["l"],
            "close": item["c"],
            "volume": item.get("v", 0)
        })
    
    return chart_data

# Generate placeholder data for demo/testing
def generate_placeholder_data(symbol, timeframe="day", days=30):
    chart_data = []
    end_time = int(time.time())
    
    # Determine time interval based on timeframe
    if timeframe == "minute":
        interval = 60
    elif timeframe == "hour":
        interval = 3600
    else:  # day, week, month
        interval = 86400
    
    base_price = 100
    if symbol.startswith("EUR"):
        base_price = 1.1
    elif symbol.startswith("GBP"):
        base_price = 1.3
    elif symbol.startswith("USD"):
        base_price = 0.9
    elif symbol.startswith("XAU"):
        base_price = 2000
    elif symbol.startswith("XAG"):
        base_price = 25
    
    # Generate random walk data
    for i in range(days):
        time_point = end_time - (interval * (days - i))
        
        if i == 0:
            close = base_price
        else:
            # Random walk with some volatility
            change = (random.random() - 0.5) * base_price * 0.02
            close = chart_data[-1]["close"] + change
        
        # Create some reasonable OHLC spread
        high = close * (1 + random.random() * 0.01)
        low = close * (1 - random.random() * 0.01)
        open_price = low + random.random() * (high - low)
        volume = int(random.random() * 10000)
        
        chart_data.append({
            "time": time_point,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
    
    return chart_data

# Alert system
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
        
        for i, alert in enumerate(self.alerts[symbol]):
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

# Streamlit UI components
def render_tradingview_chart(symbol, chart_data, container_id, height=400):
    # Convert chart data to JSON
    chart_data_json = json.dumps(chart_data)
    
    # Generate custom chart with Lightweight Charts
    js_code = f"""
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <div id="{container_id}" style="width: 100%; height: {height}px;"></div>
    <script>
        // Function to create and update chart
        function createChart() {{
            const chartData = {chart_data_json};
            
            const chart = LightweightCharts.createChart(
                document.getElementById('{container_id}'), 
                {{
                    layout: {{
                        background: {{ color: '#222' }},
                        textColor: '#DDD',
                    }},
                    grid: {{
                        vertLines: {{ color: '#444' }},
                        horzLines: {{ color: '#444' }},
                    }},
                    crosshair: {{
                        mode: LightweightCharts.CrosshairMode.Normal,
                    }},
                    rightPriceScale: {{
                        borderColor: '#666',
                    }},
                    timeScale: {{
                        borderColor: '#666',
                    }},
                    width: document.getElementById('{container_id}').clientWidth,
                    height: {height},
                }}
            );

            // Create candlestick series
            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a', 
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a', 
                wickDownColor: '#ef5350'
            }});
            
            // Set chart data
            candleSeries.setData(chartData);
            
            // Add volume series
            const volumeSeries = chart.addHistogramSeries({{
                color: '#26a69a',
                priceFormat: {{
                    type: 'volume',
                }},
                priceScaleId: '',
                scaleMargins: {{
                    top: 0.8,
                    bottom: 0,
                }},
            }});
            
            // Format volume data
            const volumeData = chartData.map(item => ({{
                time: item.time,
                value: item.volume,
                color: item.close > item.open ? '#26a69a' : '#ef5350',
            }}));
            
            volumeSeries.setData(volumeData);
            
            // Create tooltip
            const toolTipWidth = 80;
            const toolTipHeight = 80;
            const toolTipMargin = 15;
            
            // Create and style the tooltip element
            const toolTip = document.createElement('div');
            toolTip.style = `width: ${{toolTipWidth}}px; height: ${{toolTipHeight}}px; position: absolute; display: none; padding: 8px; box-sizing: border-box; font-size: 12px; color: #fff; background-color: rgba(0, 0, 0, 0.7); border-radius: 4px; z-index: 1000; top: 12px; left: 12px; pointer-events: none;`;
            document.getElementById('{container_id}').appendChild(toolTip);
            
            // Chart mouse hover event
            chart.subscribeCrosshairMove((param) => {{
                if (!param.point || !param.time || param.point.x < 0 || param.point.x > chart.clientWidth || param.point.y < 0 || param.point.y > chart.clientHeight) {{
                    toolTip.style.display = 'none';
                    return;
                }}
                
                const dateStr = new Date(param.time * 1000).toLocaleDateString();
                const data = param.seriesData.get(candleSeries);
                
                toolTip.style.display = 'block';
                const dataPoint = param.seriesData.get(candleSeries);
                const volumePoint = param.seriesData.get(volumeSeries);
                
                let content = `<div style="font-size: 12px; margin: 4px 0px;">${{dateStr}}</div>`;
                
                if (dataPoint) {{
                    content += `
                        <div>O: ${{dataPoint.open.toFixed(2)}}</div>
                        <div>H: ${{dataPoint.high.toFixed(2)}}</div>
                        <div>L: ${{dataPoint.low.toFixed(2)}}</div>
                        <div>C: ${{dataPoint.close.toFixed(2)}}</div>
                    `;
                    
                    if (volumePoint) {{
                        content += `<div>V: ${{volumePoint.value}}</div>`;
                    }}
                }}
                
                toolTip.innerHTML = content;
                
                const coordinate = candleSeries.priceToCoordinate(dataPoint.close);
                const shiftedCoordinate = param.point.x - 50;
                
                // Position tooltip
                let left = shiftedCoordinate;
                if (left < toolTipMargin) {{
                    left = toolTipMargin;
                }}
                if (left + toolTipWidth + toolTipMargin > chart.clientWidth) {{
                    left = chart.clientWidth - toolTipWidth - toolTipMargin;
                }}
                
                let top = toolTipMargin;
                if (coordinate < toolTipHeight + toolTipMargin) {{
                    top = Math.max(0, param.point.y + toolTipMargin);
                }} else {{
                    top = param.point.y - toolTipHeight - toolTipMargin;
                }}
                
                toolTip.style.left = left + 'px';
                toolTip.style.top = top + 'px';
            }});
            
            // Handle resize
            function handleResize() {{
                chart.applyOptions({{ 
                    width: document.getElementById('{container_id}').clientWidth 
                }});
            }}
            
            window.addEventListener('resize', handleResize);
            
            // Additional chart controls - could add drawing tools, indicators, etc. here
        }}
        
        // Call function to create chart when DOM is loaded
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', createChart);
        }} else {{
            createChart();
        }}
    </script>
    """
    
    return js_code

def build_dashboard():
    st.title("Market Monitor Dashboard")
    
    # Initialize session state for last update time if it doesn't exist
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now() - timedelta(minutes=10)  # Force update on first run
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Market selector
        selected_category = st.selectbox("Select Market Category", list(MARKETS.keys()))
        available_symbols = MARKETS[selected_category]
        selected_symbol = st.selectbox("Select Symbol", available_symbols)
        
        # Chart settings
        timeframe = st.selectbox("Timeframe", ["minute", "hour", "day", "week", "month"], index=2)
        days_lookback = st.slider("Days of History", min_value=1, max_value=90, value=30)
        
        # Data refresh settings
        st.header("Data Refresh Settings")
        use_real_api = st.checkbox("Use Real API (if API Key provided)", value=True)
        refresh_minutes = st.number_input("Update Frequency (minutes)", min_value=1, max_value=360, value=83)
        refresh_seconds = refresh_minutes * 60
        
        # Manual refresh button
        if st.button("Refresh Data Now"):
            update_live_prices(available_symbols, use_real_api)
            st.session_state.last_update_time = datetime.now()
            st.success(f"Data updated at {st.session_state.last_update_time.strftime('%H:%M:%S')}")
        
        # Alert creation
        st.header("Create Alert")
        alert_price = st.number_input("Alert Price", value=float(random.randint(900, 1100))/10)
        alert_condition = st.radio("Condition", ["above", "below"])
        alert_message = st.text_input("Alert Message", value=f"{selected_symbol} {alert_condition} {alert_price}")
        
        if st.button("Add Alert"):
            alert_system.add_alert(selected_symbol, alert_condition, alert_price, alert_message)
            st.success(f"Alert added for {selected_symbol}")
        
        # Display active alerts
        st.header("Active Alerts")
        active_alerts = alert_system.get_active_alerts(selected_symbol)
        for i, alert in enumerate(active_alerts):
            st.write(f"{i+1}. {alert['condition'].title()} {alert['price']}: {alert['message']}")
        
        # Display time since last update
        time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds() / 60
        st.write(f"Last data update: {time_since_update:.1f} minutes ago")
    
    # Check if it's time to update data
    time_since_update = (datetime.now() - st.session_state.last_update_time).total_seconds()
    if time_since_update > refresh_seconds:
        update_live_prices(available_symbols, use_real_api)
        st.session_state.last_update_time = datetime.now()
    
    # Main dashboard area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main chart for selected symbol
        st.header(f"{selected_symbol} Chart")
        chart_container_id = f"chart_{selected_symbol}"
        chart_data = prepare_chart_data(selected_symbol, timeframe, days_lookback)
        
        # Create HTML component for the chart
        chart_html = render_tradingview_chart(selected_symbol, chart_data, chart_container_id, height=500)
        st.components.v1.html(chart_html, height=520)
        
        # Technical analysis indicators
        with st.expander("Technical Analysis"):
            ta_col1, ta_col2, ta_col3 = st.columns(3)
            
            with ta_col1:
                st.metric("RSI", f"{random.randint(30, 70)}", f"{random.choice(['+', '-'])}{random.randint(1, 5)}")
            
            with ta_col2:
                st.metric("MACD", f"{random.choice(['+', '-'])}{random.randint(1, 100)/100}", 
                          f"{random.choice(['+', '-'])}{random.randint(1, 20)/100}")
            
            with ta_col3:
                st.metric("Bollinger Bands", f"{random.randint(0, 100)}% B", 
                          f"{random.choice(['+', '-'])}{random.randint(1, 10)}%")
    
    with col2:
        # Market overview table
        st.header("Market Overview")
        
        # Get current data for all symbols in the selected category
        live_data = {}
        for symbol in MARKETS[selected_category]:
            # Get live data from cache or generate placeholder
            symbol_data = cache.get_live(symbol)
            if symbol_data:
                price = symbol_data["price"]
                timestamp = symbol_data["timestamp"]
            else:
                # Generate placeholder data
                base = 100
                if symbol.startswith("EUR"):
                    base = 1.1
                elif symbol.startswith("GBP"):
                    base = 1.3
                elif symbol.startswith("USD"):
                    base = 0.9
                elif symbol.startswith("XAU"):
                    base = 2000
                
                price = base * (1 + (random.random() - 0.5) * 0.01)
                timestamp = int(time.time() * 1000)
            
            # Calculate dummy change (would be real in production)
            change = (random.random() - 0.5) * 0.5  # -0.25% to +0.25%
            
            live_data[symbol] = {
                "price": price,
                "change": change,
                "timestamp": timestamp
            }
            
            # Check for any triggered alerts
            if symbol_data:
                triggered_alerts = alert_system.check_alerts(symbol, price)
                for alert in triggered_alerts:
                    st.warning(f"ALERT: {alert}")
        
        # Create dataframe for the table
        market_data = []
        for symbol, data in live_data.items():
            market_data.append({
                "Symbol": symbol,
                "Price": round(data["price"], 4),
                "Change %": round(data["change"], 2),
                "Updated": datetime.fromtimestamp(data["timestamp"]/1000).strftime('%H:%M:%S')
            })
        
        df = pd.DataFrame(market_data)
        
        # Style the dataframe
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = 'red' if val < 0 else 'green'
                return f'color: {color}'
            return ''
        
        styled_df = df.style.applymap(color_negative_red, subset=['Change %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Recent alerts
        st.header("Recent Alerts")
        all_alerts = []
        for symbol, alerts in alert_system.alerts.items():
            for alert in alerts:
                all_alerts.append({
                    "Symbol": symbol,
                    "Condition": alert["condition"],
                    "Price": alert["price"],
                    "Message": alert["message"],
                    "Created": alert["created_at"].strftime('%H:%M:%S')
                })
        
        if all_alerts:
            alerts_df = pd.DataFrame(all_alerts)
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No active alerts")
        
        events_df = pd.DataFrame(events)
        st.dataframe(events_df, use_container_width=True)
    
    # Auto-refresh based on refresh_seconds
    # Note: This will refresh the entire Streamlit app, not just the data
    if refresh_seconds > 0:
        st.empty()
        time.sleep(min(refresh_seconds, 10))  # Cap at 10 seconds to avoid blocking UI
        st.rerun()

# Run the dashboard
if __name__ == "__main__":
    build_dashboard()
        
