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


# Function to fetch economic calendar events
def fetch_economic_calendar(start_date, end_date):
    if not API_KEY:
        return []
    
    # Check rate limiter
    if not rate_limiter.check_and_increment():
        st.warning("Rate limit reached. Skipping economic calendar fetch")
        return []
    
    endpoint = f"{REST_API_BASE_URL}/calendar/economic?from={start_date}&to={end_date}&apikey={API_KEY}"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching economic calendar: {e}")
        return []

# Streamlit app
def build_dashboard():
    st.title("Market Monitor Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Market category selection
    selected_category = st.sidebar.selectbox("Select Market Category", list(MARKETS.keys()))
    selected_symbols = st.sidebar.multiselect(
        "Select Symbols",
        MARKETS[selected_category],
        default=MARKETS[selected_category][:3]
    )
    
    # Refresh rate
    refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 0, 300, 60)
    
    # Data source
    use_real_api = st.sidebar.checkbox("Use Real API (Finage)", value=False)
    
    # Update live prices
    update_live_prices(selected_symbols, use_real_api=use_real_api)
    
    # Display live prices
    st.subheader("Live Prices")
    
    live_data = []
    for symbol in selected_symbols:
        data = cache.get_live(symbol)
        if data:
            live_data.append({
                "Symbol": symbol,
                "Price": data["price"],
                "Time": datetime.fromtimestamp(data["timestamp"] / 1000).strftime("%H:%M:%S"),
                "Volume": data["volume"]
            })
    
    if live_data:
        st.dataframe(pd.DataFrame(live_data))
    
    # Economic calendar
    st.subheader("Economic Calendar")
    
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    events = fetch_economic_calendar(today, next_week) if use_real_api else []
    
    # ✅ Safe handling for events
    if isinstance(events, dict):
        events_df = pd.DataFrame([events])
    elif isinstance(events, list) and events:
        events_df = pd.DataFrame(events)
    else:
        events_df = pd.DataFrame()
    
    if not events_df.empty:
        st.dataframe(events_df)
    else:
        st.write("No events available.")
    
    # Refresh logic
    if refresh_seconds > 0:
        st.empty()
        time.sleep(min(refresh_seconds, 10))  # Cap at 10s to avoid UI freeze
        st.rerun()

if __name__ == "__main__":
    build_dashboard()
