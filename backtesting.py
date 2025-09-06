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

# Streamlit page config
st.set_page_config(page_title="Trading Strategy Backtester", page_icon="📈", layout="wide")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COT API Client
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Assets (excluding RBOB Gasoline & Heating Oil)
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

# Neural network model for exit timing
class ExitTimingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(ExitTimingModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# RVOL calculation helper
def calculate_rvol(df, volume_window=20):
    if 'volume' in df.columns:
        df = df.copy()
        df['volume_20ma'] = df['volume'].rolling(volume_window).mean()
        df['rvol'] = df['volume'] / df['volume_20ma']
    else:
        df['rvol'] = 1.0
    return df


# Fetch COT data
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get("6dca-aqww", where=where_clause, order="report_date_as_yyyy_mm_dd DESC", limit=1500)
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df.get("open_interest_all", 0), errors="coerce")
                df["commercial_net"] = pd.to_numeric(df.get("commercial_long_all", 0), errors="coerce") - \
                                       pd.to_numeric(df.get("commercial_short_all", 0), errors="coerce")
                df["non_commercial_net"] = pd.to_numeric(df.get("non_commercial_long_all", 0), errors="coerce") - \
                                           pd.to_numeric(df.get("non_commercial_short_all", 0), errors="coerce")
                return df.sort_values("report_date")
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"COT fetch error: {e}")
        attempt += 1
        time.sleep(2)
    return pd.DataFrame()

# Fetch price data (patched to include volume)
def fetch_price_data(ticker_symbol: str, start_date=None, end_date=None, period="2y", interval="1d") -> pd.DataFrame:
    tk = Ticker(ticker_symbol)
    if start_date and end_date:
        df = tk.history(start=start_date, end=end_date, interval=interval).reset_index()
    else:
        df = tk.history(period=period, interval=interval).reset_index()
    # Ensure volume exists
    if 'volume' not in df.columns:
        df['volume'] = 1
    df = df[['date', 'close', 'volume']].sort_values("date")
    df['date'] = pd.to_datetime(df['date'])
    return df

# Prepare features
def prepare_features(df):
    df_features = df.copy()
    df_features['returns'] = df_features['close'].pct_change()
    df_features['log_returns'] = np.log(df_features['close']).diff()
    df_features['rolling_mean_5'] = df_features['close'].rolling(5).mean()
    df_features['rolling_mean_20'] = df_features['close'].rolling(20).mean()
    df_features['rolling_std_5'] = df_features['close'].rolling(5).std()
    df_features['rolling_std_20'] = df_features['close'].rolling(20).std()
    df_features['momentum_5'] = df_features['close'] / df_features['close'].shift(5) - 1
    df_features['momentum_20'] = df_features['close'] / df_features['close'].shift(20) - 1
    if 'commercial_net' in df_features.columns:
        df_features['comm_net_change'] = df_features['commercial_net'].diff()
    if 'non_commercial_net' in df_features.columns:
        df_features['non_comm_net_change'] = df_features['non_commercial_net'].diff()
    df_features.dropna(inplace=True)
    return df_features

# Train exit model
def train_exit_model(df, epochs=50):
    df = prepare_features(df)
    feature_cols = ['returns','log_returns','rolling_mean_5','rolling_mean_20',
                    'rolling_std_5','rolling_std_20','momentum_5','momentum_20']
    if 'commercial_net' in df.columns:
        feature_cols.extend(['commercial_net','non_commercial_net','comm_net_change','non_comm_net_change'])
    df['future_returns_5d'] = df['close'].pct_change(5).shift(-5)
    df.dropna(inplace=True)
    X = df[feature_cols].values
    y = df['future_returns_5d'].values.reshape(-1,1)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                                            torch.tensor(y_train,dtype=torch.float32)), batch_size=32, shuffle=True)
    model = ExitTimingModel(input_size=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model, scaler_X, scaler_y

# Backtest strategy with NN exit timing
def backtest_strategy_with_nn(price_df, cot_df=None, exit_model=None, feature_scaler=None, target_scaler=None, exit_days=5):
    df = price_df.copy()
    if cot_df is not None and not cot_df.empty:
        df = df.merge(cot_df[['report_date','commercial_net','non_commercial_net']],
                      left_on='date', right_on='report_date', how='left')
        df.fillna(method='ffill', inplace=True)
    df = calculate_rvol(df)
    # placeholder signals
    df['extension'] = 0.0
    df['health_gauge'] = 5.0
    df['buy_signal'] = (df['health_gauge']>=7) & (df['rvol']>=1.5) & (df['extension']<0)
    df['sell_signal'] = (df['health_gauge']<=3) & (df['rvol']>=1.5) & (df['extension']>0)
    df['strong_buy'] = df['buy_signal'] & (df['rvol']>2.0) & (df['extension']<-0.5)
    df['strong_sell'] = df['sell_signal'] & (df['rvol']>2.0) & (df['extension']>0.5)
    df['position'] = 0
    exit_timer = 0
    current_position = 0
    for i in range(1,len(df)):
        # Entry
        if df['strong_buy'].iloc[i] and current_position<=0:
            current_position = 1
            exit_timer = 0
        elif df['strong_sell'].iloc[i] and current_position>=0:
            current_position = -1
            exit_timer = 0
        elif df['buy_signal'].iloc[i] and current_position<=0 and not df['strong_sell'].iloc[i]:
            current_position = 1
            exit_timer = 0
        elif df['sell_signal'].iloc[i] and current_position>=0 and not df['strong_buy'].iloc[i]:
            current_position = -1
            exit_timer = 0
        # NN exit
        if exit_model is not None and i>20:
            feature_cols = ['returns','log_returns','rolling_mean_5','rolling_mean_20',
                            'rolling_std_5','rolling_std_20','momentum_5','momentum_20']
            if 'commercial_net' in df.columns:
                feature_cols.extend(['commercial_net','non_commercial_net','comm_net_change','non_comm_net_change'])
            feature_cols.extend(['position','buy_signal','sell_signal'])
            try:
                features = df.iloc[i][feature_cols].values.reshape(1,-1)
                features_scaled = feature_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled,dtype=torch.float32)
                with torch.no_grad():
                    pred_scaled = exit_model(features_tensor).numpy()
                    pred = target_scaler.inverse_transform(pred_scaled)[0][0]
                if (current_position>0 and pred<-0.005) or (current_position<0 and pred>0.005):
                    if exit_timer==0: exit_timer=exit_days
            except:
                pass
        # Countdown exit
        if exit_timer>0:
            exit_timer-=1
            if exit_timer==0:
                current_position=0
        df.loc[df.index[i],'position'] = current_position
    df['returns'] = df['close'].pct_change()*df['position'].shift(1)
    df['total'] = (1+df['returns'].fillna(0)).cumprod()
    return df

# Main
def main():
    st.title("Backtester")
    asset_choice = st.selectbox("Select Asset", list(assets.keys()))
    price_df = fetch_price_data(assets[asset_choice])
    cot_df = fetch_cot_data(asset_choice)
    exit_model, scaler_X, scaler_y = train_exit_model(price_df)
    backtest_results = backtest_strategy_with_nn(price_df, cot_df, exit_model, scaler_X, scaler_y)
    st.line_chart(backtest_results[['close','total']])

if __name__=="__main__":
    main()
