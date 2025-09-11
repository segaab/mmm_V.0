###############################################################################
# train_mispricing_nn.py
###############################################################################
"""
End-to-end training pipeline:
FinBERT + structured features → NN → back-test metrics driven selection
"""
import os, warnings, math, json, time, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from yahooquery import Ticker
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###############################################################################
# 🔑 API KEYS
###############################################################################
NEWSAPI_KEY = "3ca6e30c7f8348c0b5feb83ad29eb76b"   # <-- insert your key

###############################################################################
# 📥 DATA DOWNLOAD
###############################################################################
def fetch_price(sym, start, end, interval="1d"):
    t = Ticker(sym, timeout=30)
    delta = (end - start).days + 2
    hist = t.history(period=f"{delta}d", interval=interval)
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index()
    hist.rename(columns={"date": "datetime"}, inplace=True)
    hist["datetime"] = pd.to_datetime(hist["datetime"])
    hist.set_index("datetime", inplace=True)
    return hist[["close", "volume"]]

def fetch_sentiment(keyword, start, end):
    api = NewsApiClient(api_key=NEWSAPI_KEY)
    arts = api.get_everything(q=keyword,
                              from_param=start.date().isoformat(),
                              to=end.date().isoformat(),
                              sort_by="publishedAt",
                              language="en",
                              page_size=100)
    model_name = "ProsusAI/finbert"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sents = []
    for a in arts.get("articles", []):
        ts   = pd.to_datetime(a["publishedAt"])
        txt  = a["title"][:512]
        tok_out = tok(txt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**tok_out).logits
        probs = torch.softmax(logits, -1).squeeze().numpy()
        score = probs[2] - probs[0]        # pos – neg
        sents.append({"datetime": ts, "sent": score})
    df = pd.DataFrame(sents)
    if df.empty:
        return pd.DataFrame()
    return df.set_index("datetime").resample("1D").mean()

###############################################################################
# 🛠️  FEATURE ENGINEERING
###############################################################################
def build_features(price_df, factor_df, sent_df):
    df = pd.concat({"price": price_df["close"],
                    "factor": factor_df["close"].reindex(price_df.index),
                    "volume": price_df["volume"]}, axis=1).ffill()

    # technicals
    df["ret_1"]  = df.price.pct_change()
    df["ret_5"]  = df.price.pct_change(5)
    df["vol_chg"]= df.volume.pct_change()
    df["ma_10"]  = df.price.rolling(10).mean() / df.price - 1
    df["ma_30"]  = df.price.rolling(30).mean() / df.price - 1
    df["factor_ret"] = df.factor.pct_change()

    # sentiment
    df["sent"] = sent_df.sent.reindex(df.index).ffill().fillna(0)

    # target: next-day return
    df["target"] = df.price.pct_change().shift(-1)

    return df.dropna()

###############################################################################
# 📦 DATASET
###############################################################################
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1,1)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]

###############################################################################
# 🧠 MODEL
###############################################################################
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
# 💹 BACKTEST HELPERS  (from user’s spec)
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
# 🏃 WALK-FORWARD TRAIN / VALIDATE
###############################################################################
def walk_forward(df, train_years=1, test_months=3,
                 epochs=200, patience=10,
                 lr=1e-3, threshold=0.0):
    results=[]; best_global=None; best_state=None
    scaler = StandardScaler()
    features = df.drop(columns=["target"])
    for start in pd.date_range(df.index.min(),
                               df.index.max()-pd.DateOffset(years=train_years),
                               freq=f"{test_months}M"):
        train_end = start + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(months=test_months)
        train_df  = df.loc[start:train_end]
        test_df   = df.loc[train_end:test_end]
        if len(test_df)<30: break

        X_train = scaler.fit_transform(train_df.drop(columns=["target"]))
        y_train = train_df.target.values
        X_test  = scaler.transform(test_df.drop(columns=["target"]))
        y_test  = test_df.target.values

        train_ds=TabDataset(X_train,y_train)
        test_ds =TabDataset(X_test, y_test)
        tr_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        te_dl = DataLoader(test_ds,  batch_size=256)

        model=MLP(X_train.shape[1])
        opt=torch.optim.Adam(model.parameters(), lr=lr)
        best_sharpe=-9; epochs_since=0

        for ep in range(epochs):
            model.train()
            for xb,yb in tr_dl:
                pred=model(xb)
                loss=nn.functional.mse_loss(pred,yb)
                loss.backward(); opt.step(); opt.zero_grad()

            # ---- validation → economic metric
            model.eval()
            with torch.no_grad():
                val_pred = []
                for xb,_ in te_dl:
                    val_pred.append(model(xb).squeeze().cpu().numpy())
                val_pred = np.concatenate(val_pred)
            sig = nn_to_signal(val_pred, threshold)
            equity,pnl,rr = backtest_nn(test_df.price, sig)
            m = strategy_metrics(equity,pnl,rr)
            cur_sharpe=m["Sharpe"]

            if cur_sharpe > best_sharpe:
                best_sharpe=cur_sharpe
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                epochs_since=0
            else:
                epochs_since+=1
            if epochs_since>=patience:
                break  # early stop

        results.append({"fold_start":start, "Sharpe":best_sharpe,
                        "WinRate":m["WinRate"]})
        # keep global best
        if (best_global is None) or (best_sharpe>best_global["Sharpe"]):
            best_global={"Sharpe":best_sharpe, "fold_start":start}
            global_state=best_state

    return results, global_state, scaler

###############################################################################
# 🚀 MAIN
###############################################################################
if __name__=="__main__":
    SYMBOL   = "GC=F"        # Gold futures
    FACTOR   = "DX-Y"        # Dollar index
    START    = datetime(2014,1,1)
    END      = datetime.today()

    print("Downloading data …")
    price = fetch_price(SYMBOL, START, END)
    factor= fetch_price(FACTOR, START, END)
    sent  = fetch_sentiment("gold", START, END)

    df = build_features(price, factor, sent)

    print("Walk-forward training …")
    res, state, scaler = walk_forward(df,
                                      train_years=2,
                                      test_months=3,
                                      epochs=300,
                                      patience=15,
                                      lr=1e-3,
                                      threshold=0.0002)

    print(pd.DataFrame(res))
    print(f"Best global Sharpe: {max(r['Sharpe'] for r in res):.2f}")

    # save artefacts
    torch.save({"state_dict":state,
                "scaler_mean":scaler.mean_,
                "scaler_scale":scaler.scale_,
                "features":df.drop(columns=["target"]).columns.tolist()},
               "best_model.pt")
    print("Model saved → best_model.pt")
###############################################################################
# end of file
###############################################################################
