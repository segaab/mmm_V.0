###############################################################################
# app_nn_backtester.py – Streamlit Mis-Pricing + NN Back-Tester
###############################################################################
import os, json, time, math, warnings, random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from yahooquery import Ticker

import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NN Mis-Pricing Back-Tester", layout="wide")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###############################################################################
# 🔑 API KEYS
###############################################################################
FINAGE_API_KEY = "API_KEY39A0ZSC2ZV7VMRZLROAG2D712IQK260L"
NEWSAPI_KEY   = "3ca6e30c7f8348c0b5feb83ad29eb76b"

###############################################################################
# 🔗 Factor Map Loader
###############################################################################
def load_factor_library():
    factor_map = {
        "EURUSD=X": "DX-Y",
        "GBPUSD=X": "DX-Y",
        "GC=F":     "DX-Y",
        "CL=F":     "^VIX",
        "^GSPC":    "^VIX",
        "DX-Y":     "GC=F"
    }
    return factor_map

FACTOR_MAP = load_factor_library()

###############################################################################
# 🔗 YahooQuery wrapper
###############################################################################
@st.cache_data(show_spinner="Downloading price data …")
def fetch_history(ticker: str, start: datetime, end: datetime,
                  tf: str = "1d") -> pd.DataFrame:
    delta = (end - start).days + 2
    t = Ticker(ticker, timeout=30)
    hist = t.history(period=f"{delta}d", interval=tf)
    if hist.empty:
        return pd.DataFrame()
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index()
    hist = hist.rename(columns={"date": "datetime"})
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True)
    hist = hist.set_index("datetime")
    hist = hist[~hist.index.duplicated(keep="first")]
    hist = hist.sort_index()
    return hist[["open", "high", "low", "close", "volume"]]

###############################################################################
# 🔗 FinBERT Sentiment fetcher
###############################################################################
@st.cache_data(show_spinner="Fetching FinBERT sentiment …")
def fetch_sentiment_finbert(keyword: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        from newsapi import NewsApiClient
        api = NewsApiClient(api_key=NEWSAPI_KEY)
        arts = api.get_everything(q=keyword,
                                  from_param=start.date().isoformat(),
                                  to=end.date().isoformat(),
                                  sort_by="relevancy",
                                  language="en",
                                  page_size=100)
        model_name = "ProsusAI/finbert"
        tok   = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sents = []
        for a in arts.get("articles", []):
            ts = pd.to_datetime(a["publishedAt"])
            txt = a["title"][:512]
            tok_out = tok(txt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model(**tok_out).logits
            probs = torch.softmax(logits, -1).squeeze().numpy()
            score = probs[2] - probs[0]        # pos – neg
            sents.append({"datetime": ts, "sent": score})
        df = pd.DataFrame(sents)
        return df.set_index("datetime").resample("1D").mean()
    except Exception as e:
        st.warning(f"FinBERT sentiment error → {e}")
        return pd.DataFrame()

###############################################################################
# 🔗 Feature builder
###############################################################################
@st.cache_data(show_spinner="Building feature dataframe …")
def build_feature_df(price_df: pd.DataFrame,
                     factor_map: dict,
                     symbol: str,
                     start: datetime,
                     end: datetime,
                     use_news: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=price_df.index)
    df["price"] = price_df["close"]
    df["volume"] = price_df["volume"]
    df["ret_1"] = df["price"].pct_change()
    df["ret_5"] = df["price"].pct_change(5)
    df["ma_10"] = df["price"].rolling(10).mean()/df["price"] - 1
    df["ma_30"] = df["price"].rolling(30).mean()/df["price"] - 1
    df["vol_chg"] = df["volume"].pct_change()

    # factor features
    factor_sym = factor_map.get(symbol)
    if factor_sym:
        factor_df = fetch_history(factor_sym, start, end, "1d")
        df["factor"] = factor_df["close"].reindex(df.index)
        df["factor_ret"] = df["factor"].pct_change()
    else:
        df["factor"] = 0
        df["factor_ret"] = 0

    # sentiment
    if use_news:
        news_df = fetch_sentiment_finbert(symbol.split()[0], start, end)
        df["sent"] = news_df["sent"].reindex(df.index).ffill().fillna(0)
    else:
        df["sent"] = 0

    df["target"] = df["price"].pct_change().shift(-1)
    return df.dropna()



###############################################################################
# 🗂 Dataset & Neural Network
###############################################################################
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self,x): return self.net(x)

###############################################################################
# 🧮 Backtest helpers
###############################################################################
def nn_to_signal(preds, threshold=0.0):
    sig = np.zeros_like(preds)
    sig[preds > threshold]  = 1
    sig[preds < -threshold] = -1
    return sig

def backtest_nn(price, signals, capital=1e6,
                cost_per_unit=0.0005, max_holding=20):
    price = price.ffill()
    returns = price.pct_change().shift(-1)
    pos = signals.copy()
    equity=[capital]; pnl_list=[]; rr=[]
    cur=0; hold=0
    for i in range(len(pos)):
        cost = abs(pos[i]-cur)*cost_per_unit*equity[-1]
        ret  = returns.iloc[i]*cur*equity[-1]
        equity.append(equity[-1]+ret-cost)
        pnl_list.append(ret-cost)
        if pos[i]!=0: hold+=1
        else: hold=0
        if hold>=max_holding:
            cur=0; hold=0
        else:
            cur=pos[i]
        if cost>0:
            rr.append((ret-cost)/cost)
    equity = pd.Series(equity[1:], index=price.index)
    pnl    = pd.Series(pnl_list, index=price.index)
    return equity, pnl, rr

def strategy_metrics(equity, pnl, rr):
    daily_ret = pnl / equity.shift(1)
    sharpe = np.sqrt(252)*daily_ret.mean()/daily_ret.std() if daily_ret.std()>0 else -9
    wins   = (pnl>0).sum(); total=len(pnl[pnl!=0])
    winrate= wins/total*100 if total>0 else 0
    rr_arr = np.array(rr)
    return {"Sharpe": sharpe,
            "WinRate": winrate,
            "RRmean": rr_arr.mean() if len(rr_arr)>0 else np.nan}

###############################################################################
# 🚀 Walk-forward training & selection
###############################################################################
def walk_forward(df, train_years=1, test_months=3,
                 epochs=300, patience=15,
                 lr=1e-3, threshold=0.0):
    results=[]; best_global=None; best_state=None
    scaler_mean = df.drop(columns=["target"]).mean()
    scaler_std  = df.drop(columns=["target"]).std()
    X_all = ((df.drop(columns=["target"]) - scaler_mean)/scaler_std).values
    y_all = df["target"].values

    idx = df.index
    for start in pd.date_range(df.index.min(),
                               df.index.max()-pd.DateOffset(years=train_years),
                               freq=f"{test_months}M"):
        train_end = start + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(months=test_months)
        train_mask = (idx>=start) & (idx<=train_end)
        test_mask  = (idx>train_end) & (idx<=test_end)
        if test_mask.sum()<30: continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test   = X_all[test_mask], y_all[test_mask]
        train_ds = TabDataset(X_train, y_train)
        test_ds  = TabDataset(X_test, y_test)
        tr_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        te_dl = DataLoader(test_ds, batch_size=256)

        model = MLP(X_train.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_sharpe = -9
        epochs_since = 0

        for ep in range(epochs):
            model.train()
            for xb, yb in tr_dl:
                pred = model(xb)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward(); opt.step(); opt.zero_grad()

            # ---- validation
            model.eval()
            with torch.no_grad():
                val_pred = []
                for xb,_ in te_dl:
                    val_pred.append(model(xb).squeeze().cpu().numpy())
                val_pred = np.concatenate(val_pred)
            sig = nn_to_signal(val_pred, threshold)
            equity, pnl, rr = backtest_nn(df["price"][test_mask], sig)
            m = strategy_metrics(equity, pnl, rr)
            cur_sharpe = m["Sharpe"]

            if cur_sharpe>best_sharpe:
                best_sharpe=cur_sharpe
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                epochs_since=0
            else:
                epochs_since+=1
            if epochs_since>=patience:
                break

        results.append({"fold_start":start, "Sharpe":best_sharpe,
                        "WinRate":m["WinRate"]})
        if (best_global is None) or (best_sharpe>best_global["Sharpe"]):
            best_global={"Sharpe":best_sharpe, "fold_start":start}
            global_state = best_state

    return results, global_state, scaler_mean, scaler_std




###############################################################################
# 🎛️ Streamlit UI – NN Backtester
###############################################################################
st.sidebar.header("🤖 Neural Network Parameters")
use_nn    = st.sidebar.checkbox("Enable NN Mis-Pricing", value=True)
train_nn  = st.sidebar.checkbox("Train NN (walk-forward)", value=True)
nn_epochs = st.sidebar.slider("NN Epochs", 200, 500, 300, 50)
nn_patience = st.sidebar.slider("Early Stop Patience", 5, 50, 15, 1)
threshold = st.sidebar.slider("Signal Threshold", 0.0, 0.01, 0.0002, 0.0001)

run_nn_btn = st.sidebar.button("🚀 Run NN Back-Test")

if run_nn_btn:
    with st.spinner("Fetching data …"):
        price_df = fetch_history(symbol, start_date, end_date, interval)
        if price_df.empty:
            st.error("No price data found."); st.stop()

        factor_sym = FACTOR_MAP.get(symbol)
        factor_df = fetch_history(factor_sym, start_date, end_date, interval) if factor_sym else pd.DataFrame()
        news_df = fetch_sentiment(asset_name.split()[0], start_date, end_date) if use_news else pd.DataFrame()

        # build features
        df = build_features(price_df, factor_df, news_df)

    # ----- Walk-forward training & backtest -----
    if train_nn:
        st.info("Training NN with walk-forward validation …")
        results, state_dict, scaler_mean, scaler_std = walk_forward(
            df,
            train_years=2,
            test_months=3,
            epochs=nn_epochs,
            patience=nn_patience,
            threshold=threshold
        )
        st.success("Training complete!")

        # save best model
        torch.save({
            "state_dict": state_dict,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "features": df.drop(columns=["target"]).columns.tolist()
        }, "best_nn_model.pt")
        st.info("Saved best NN model → best_nn_model.pt")

        st.subheader("Fold Results")
        st.dataframe(pd.DataFrame(results))

    # ----- Load best model for plotting -----
    ART = torch.load("best_nn_model.pt", map_location="cpu")
    feat_cols = ART["features"]
    model = MLP(len(feat_cols))
    model.load_state_dict(ART["state_dict"])
    model.eval()
    X_live = ((df[feat_cols]-ART["scaler_mean"])/ART["scaler_std"]).values.astype(np.float32)
    with torch.no_grad():
        pred = model(torch.tensor(X_live)).squeeze().numpy()
    nn_signal = nn_to_signal(pred, threshold)

    equity, pnl, rr = backtest_nn(df["price"], nn_signal)
    stats = strategy_metrics(equity, pnl, rr)

    # ===== Plots =====
    col1, col2 = st.columns([3,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines", name=symbol))
        entries = df.index[(nn_signal==1)]
        exits   = df.index[(nn_signal==-1)]
        fig.add_trace(go.Scatter(x=entries, y=df["price"].loc[entries],
                                 mode="markers", name="Long Entries",
                                 marker=dict(color="green", size=8, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=exits, y=df["price"].loc[exits],
                                 mode="markers", name="Short Entries",
                                 marker=dict(color="red", size=8, symbol="triangle-down")))
        fig.update_layout(height=500, title="Price & NN Signals")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=equity.index, y=equity,
                                  mode="lines", name="Equity"))
        fig2.update_layout(height=250, title="Equity Curve")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("NN Performance")
        st.metric("Sharpe",  f"{stats['Sharpe']:.2f}")
        st.metric("WinRate %",f"{stats['WinRate']:.1f}")
        st.metric("RR Mean", f"{stats['RRmean']:.4f}")

else:
    st.info("Use the sidebar to configure parameters and run the NN Back-Test.")