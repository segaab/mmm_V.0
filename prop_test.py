def execute_backtest_with_profile(signals_df: pd.DataFrame, asset_name: str, risk_profile: dict,
starting_balance=10000, lot_size=1.0, stop_loss_pips=50):
if signals_df.empty:
return pd.DataFrame(), pd.DataFrame(), {}

balance = starting_balance  
equity_curve = []  
trades = []  
  
# Challenge tracking  
current_phase = 1  
phase_start_balance = starting_balance  
daily_starting_balance = starting_balance  
current_date = None  
max_drawdown = 0  
peak_balance = starting_balance  
trading_days = 0  
daily_profit = 0  
total_phase_profit = 0  
  
# Challenge targets  
phase_targets = [  
    risk_profile.get("phase_1_target", 0.08),  
    risk_profile.get("phase_2_target", 0.05),  
    risk_profile.get("phase_3_target", None)  
]  
  
challenge_status = "Phase 1"  
is_funded = False  

for i in range(1, len(signals_df)):  
    current_row = signals_df.iloc[i]  
      
    # Check for new trading day  
    if current_row["date"].date() != current_date:  
        if current_date is not None:  
            trading_days += 1  
        daily_starting_balance = balance  
        daily_profit = 0  
        current_date = current_row["date"].date()  
          
    # Daily loss limit check  
    daily_loss_pct = (daily_starting_balance - balance) / daily_starting_balance  
    if daily_loss_pct >= risk_profile["daily_drawdown_limit"]:  
        challenge_status = "FAILED - Daily Loss Limit Exceeded"  
        break  
          
    # Maximum drawdown check  
    if balance > peak_balance:  
        peak_balance = balance  
    current_drawdown = (peak_balance - balance) / peak_balance  
    if current_drawdown > max_drawdown:  
        max_drawdown = current_drawdown  
    if current_drawdown >= risk_profile["max_drawdown_limit"]:  
        challenge_status = "FAILED - Maximum Drawdown Exceeded"  
        break  

    # Check phase completion  
    phase_profit_pct = (balance - phase_start_balance) / phase_start_balance  
    current_target = phase_targets[current_phase - 1] if current_phase <= len(phase_targets) else None  
      
    if current_target and phase_profit_pct >= current_target:  
        if current_phase == len([t for t in phase_targets if t is not None]):  
            challenge_status = "FUNDED"  
            is_funded = True  
        else:  
            current_phase += 1  
            phase_start_balance = balance  
            challenge_status = f"Phase {current_phase}"  

    signal = signals_df.iloc[i-1].get("signal", 0)  
    price_open = current_row.get("close", np.nan)  
    price_prev = signals_df.iloc[i-1].get("close", np.nan)  

    if signal != 0 and not np.isnan(price_open) and not np.isnan(price_prev):  
        # Calculate position size using risk profile  
        position_lots = calculate_position_size_with_profile(  
            account_balance=balance,  
            risk_profile=risk_profile,  
            asset_name=asset_name,  
            price=price_open,  
            stop_loss_pips=stop_loss_pips  
        ) * lot_size  

        if position_lots > 0:  
            # Calculate trade return  
            price_change_pct = (price_open - price_prev) / price_prev  
            leverage = risk_profile["leverage_limits"].get(asset_classes.get(asset_name, ("FX", "default"))[0], 10)  
            trade_return_pct = price_change_pct * leverage * signal * position_lots * 0.01  
              
            # Apply consistency rule for TopOneTrader  
            if "consistency_rule" in risk_profile:  
                max_daily_target = current_target * risk_profile["consistency_rule"] if current_target else 0.05  
                if abs(trade_return_pct) > max_daily_target:  
                    trade_return_pct = max_daily_target if trade_return_pct > 0 else -max_daily_target  
              
            trade_return_amount = balance * trade_return_pct  
            new_balance = balance + trade_return_amount  
              
            # Check if trade would breach daily limit  
            new_daily_loss = (daily_starting_balance - new_balance) / daily_starting_balance  
            if new_daily_loss < risk_profile["daily_drawdown_limit"]:  
                balance = new_balance  
                daily_profit += trade_return_amount  
                total_phase_profit += trade_return_amount  
                  
                trades.append({  
                    "date": current_row["date"],  
                    "signal": signal,  
                    "price": price_open,  
                    "position_lots": position_lots,  
                    "trade_return_pct": trade_return_pct,  
                    "trade_return_amount": trade_return_amount,  
                    "balance": balance,  
                    "phase": current_phase,  
                    "drawdown": current_drawdown,  
                    "daily_loss": daily_loss_pct,  
                    "phase_progress": phase_profit_pct  
                })  

    equity_curve.append({  
        "date": current_row["date"],   
        "balance": balance,  
        "drawdown": current_drawdown,  
        "daily_loss": daily_loss_pct,  
        "phase": current_phase,  
        "phase_progress": phase_profit_pct,  
        "challenge_status": challenge_status,  
        "target": current_target  
    })  

equity_df = pd.DataFrame(equity_curve)  
trades_df = pd.DataFrame(trades)  

# Calculate performance metrics  
wins = trades_df[trades_df["trade_return_amount"] > 0].shape[0] if not trades_df.empty else 0  
losses = trades_df[trades_df["trade_return_amount"] <= 0].shape[0] if not trades_df.empty else 0  
win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0  
  
total_return = (balance - starting_balance) / starting_balance  
  
metrics = {  
    "final_balance": balance,  
    "total_return": total_return,  
    "total_return_pct": total_return * 100,  
    "num_trades": len(trades_df),  
    "wins": wins,  
    "losses": losses,  
    "win_rate": win_rate * 100,  
    "max_drawdown": max_drawdown * 100,  
    "max_daily_loss": equity_df["daily_loss"].max() * 100 if not equity_df.empty else 0,  
    "trading_days": trading_days,  
    "current_phase": current_phase,  
    "challenge_status": challenge_status,  
    "is_funded": is_funded,  
    "phase_progress": equity_df["phase_progress"].iloc[-1] * 100 if not equity_df.empty else 0  
}  

return equity_df, trades_df, metrics

--- Helper Functions ---

def load_price_data(asset_name, years_back):
ticker = assets.get(asset_name)
if ticker is None:
logger.warning(f"No ticker mapping found for {asset_name}.")
return pd.DataFrame()
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=years_back * 365)
return fetch_price_data_yahoo(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

def load_cot_data(asset_name):
return fetch_cot_data(asset_name)

def process_data_for_backtesting(asset_name, years_back):
price_df = load_price_data(asset_name, years_back)
cot_df = load_cot_data(asset_name)

if price_df.empty or cot_df.empty:  
    return pd.DataFrame()  

price_df = calculate_rvol(price_df)  
merged_df = merge_cot_price(cot_df, price_df)  
signals_df = generate_signals(merged_df)  
return signals_df

def display_prop_firm_header():
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
<h1 style='color: white; margin: 0; font-size: 2.5rem;'>PropFirm Challenge Backtester</h1>
<p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Professional Trading Performance Analysis</p>
</div>
""", unsafe_allow_html=True)

def display_challenge_status(metrics, risk_profile):
status = metrics.get("challenge_status", "Phase 1")
phase = metrics.get("current_phase", 1)
progress = metrics.get("phase_progress", 0)

# Determine status color  
if "FAILED" in status:  
    status_color = "#ff4757"  
elif "FUNDED" in status:  
    status_color = "#2ed573"  
else:  
    status_color = "#3742fa"  
  
# Current phase target  
if phase == 1:  
    target = risk_profile.get("phase_1_target", 0.08) * 100  
elif phase == 2:  
    target = risk_profile.get("phase_2_target", 0.05) * 100  
else:  
    target = risk_profile.get("phase_3_target", 0.06) * 100 if risk_profile.get("phase_3_target") else 0  
  
st.markdown(f"""  
<div style='background: {status_color}; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;'>  
    <h2 style='color: white; margin: 0;'>{status}</h2>  
    <p style='color: white; margin: 0.5rem 0 0 0;'>Progress: {progress:.1f}% / Target: {target:.1f}%</p>  
</div>  
""", unsafe_allow_html=True)

def create_prop_metrics_dashboard(metrics, risk_profile):
# Main metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:  
    profit_color = "profit-positive" if metrics["total_return"] > 0 else "profit-negative"  
    st.markdown(f"""  
    <div class="metric-card {profit_color}">  
        <h3>Total P&L</h3>  
        <h2>${metrics["final_balance"] - 10000:,.2f}</h2>  
        <p>{metrics["total_return_pct"]:.2f}%</p>  
    </div>  
    """, unsafe_allow_html=True)  
  
with col2:  
    st.markdown(f"""  
    <div class="metric-card">  
        <h3>Max Drawdown</h3>  
        <h2>{metrics["max_drawdown"]:.2f}%</h2>  
        <p>Limit: {risk_profile["max_drawdown_limit"]*100:.0f}%</p>  
    </div>  
    """, unsafe_allow_html=True)  
  
with col3:  
    st.markdown(f"""  
    <div class="metric-card">  
        <h3>Win Rate</h3>  
        <h2>{metrics["win_rate"]:.1f}%</h2>  
        <p>{metrics["wins"]}W / {metrics["losses"]}L</p>  
    </div>  
    """, unsafe_allow_html=True)  
  
with col4:  
    st.markdown(f"""  
    <div class="metric-card">  
        <h3>Trading Days</h3>  
        <h2>{metrics["trading_days"]}</h2>  
        <p>Min: {risk_profile["min_trading_days"]}</p>  
    </div>  
    """, unsafe_allow_html=True)

def create_phase_progress_chart(equity_df, risk_profile):
if equity_df.empty:
return

fig = make_subplots(  
    rows=2, cols=1,  
    subplot_titles=("Account Balance & Phase Progress", "Drawdown"),  
    vertical_spacing=0.1  
)  
  
# Balance line  
fig.add_trace(  
    go.Scatter(  
        x=equity_df["date"],  
        y=equity_df["balance"],  
        mode="lines",  
        name="Balance",  
        line=dict(color="#2ed573", width=2)  
    ),  
    row=1, col=1  
)  
  
# Phase targets  
if not equity_df.empty:  
    starting_balance = equity_df["balance"].iloc[0]  
      
    # Phase 1 target line  
    phase1_target = starting_balance * (1 + risk_profile.get("phase_1_target", 0.08))  
    fig.add_hline(  
        y=phase1_target,  
        line_dash="dash",  
        line_color="#ff6b6b",  
        annotation_text="Phase 1 Target",  
        row=1, col=1  
    )  
      
    # Phase 2 target line  
    if risk_profile.get("phase_2_target"):  
        phase2_target = phase1_target * (1 + risk_profile["phase_2_target"])  
        fig.add_hline(  
            y=phase2_target,  
            line_dash="dash",  
            line_color="#4ecdc4",  
            annotation_text="Phase 2 Target",  
            row=1, col=1  
        )  
  
# Drawdown  
fig.add_trace(  
    go.Scatter(  
        x=equity_df["date"],  
        y=equity_df["drawdown"] * 100,  
        mode="lines",  
        name="Drawdown %",  
        line=dict(color="#ff4757", width=2),  
        fill="tonexty"  
    ),  
    row=2, col=1  
)  
  
# Drawdown limit line  
fig.add_hline(  
    y=risk_profile["max_drawdown_limit"] * 100,  
    line_dash="dash",  
    line_color="#ff4757",  
    annotation_text="Max DD Limit",  
    row=2, col=1  
)  
  
fig.update_layout(  
    height=600,  
    showlegend=True,  
    title="PropFirm Challenge Performance"  
)  
  
st.plotly_chart(fig, use_container_width=True)

--- Main Application ---

def main():
display_prop_firm_header()

# Sidebar for parameters  
with st.sidebar:  
    st.header("🎯 Challenge Parameters")  
      
    # Risk Profile Selection  
    selected_profile = st.selectbox(  
        "Select Prop Firm Challenge",  
        list(RISK_PROFILES.keys()),  
        index=0  
    )  
    risk_profile = RISK_PROFILES[selected_profile]  
      
    st.markdown("---")  
      
    # Display selected profile info  
    st.markdown(f"""  
    **{risk_profile['name']} Rules:**  
    - Daily DD Limit: {risk_profile['daily_drawdown_limit']*100:.0f}%  
    - Max DD Limit: {risk_profile['max_drawdown_limit']*100:.0f}%  
    - Phase 1 Target: {risk_profile['phase_1_target']*100:.0f}%  
    - Phase 2 Target: {risk_profile['phase_2_target']*100:.0f}%  
    """)  
      
    if risk_profile.get('phase_3_target'):  
        st.markdown(f"- Phase 3 Target: {risk_profile['phase_3_target']*100:.0f}%")  
      
    st.markdown("---")  

    # Trading Parameters  
    selected_asset = st.selectbox("Select Asset", list(assets.keys()), index=0)  
    years_back = st.slider("Years to Backtest", min_value=1, max_value=10, value=3)  
      
    st.markdown("**Strategy Parameters**")  
    buy_thresh = st.number_input("Buy Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05)  
    sell_thresh = st.number_input("Sell Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)  
      
    st.markdown("**Risk Parameters**")  
    lot_size = st.number_input("Lot Size Multiplier", min_value=0.01, value=1.0, step=0.01)  
    stop_loss_pips = st.number_input("Stop Loss (pips)", min_value=1, value=50, step=1)  
    starting_balance = st.number_input("Starting Balance", min_value=1000, value=10000, step=1000)  

# Main content area  
with st.spinner("🔄 Loading and processing market data..."):  
    # Load and process data  
    price_df = load_price_data(selected_asset, years_back)  
    cot_df = load_cot_data(selected_asset)  

    if price_df.empty or cot_df.empty:  
        st.error(f"❌ No data available for {selected_asset}. Please try another asset.")  
        return  

    price_df = calculate_rvol(price_df)  
    signals_df = merge_cot_price(cot_df, price_df)  
    signals_df = generate_signals(signals_df, buy_threshold=buy_thresh, sell_threshold=sell_thresh)  

# Execute backtest with risk profile  
equity_df, trades_df, metrics = execute_backtest_with_profile(  
    signals_df,  
    asset_name=selected_asset,  
    risk_profile=risk_profile,  
    starting_balance=starting_balance,  
    lot_size=lot_size,  
    stop_loss_pips=stop_loss_pips  
)  

# Display challenge status  
display_challenge_status(metrics, risk_profile)  
  
# Display metrics dashboard  
create_prop_metrics_dashboard(metrics, risk_profile)  
  
# Performance charts  
st.subheader("📊 Performance Analysis")  
create_phase_progress_chart(equity_df, risk_profile)  
  
# Detailed metrics  
col1, col2 = st.columns(2)  
  
with col1:  
    st.subheader("📈 Challenge Metrics")  
    if metrics["is_funded"]:  
        st.success("🎉 CHALLENGE PASSED - FUNDED!")  
    elif "FAILED" in metrics["challenge_status"]:  
        st.error(f"❌ {metrics['challenge_status']}")  
    else:  
        st.info(f"⏳ In Progress - {metrics['challenge_status']}")  
          
    st.metric("Current Phase", metrics["current_phase"])  
    st.metric("Phase Progress", f"{metrics['phase_progress']:.2f}%")  
    st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")  
    st.metric("Max Daily Loss", f"{metrics['max_daily_loss']:.2f}%")  
  
with col2:  
    st.subheader("📊 Trading Statistics")  
    st.metric("Total Trades", metrics["num_trades"])  
    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")  
    st.metric("Wins", metrics["wins"])  
    st.metric("Losses", metrics["losses"])  
    st.metric("Trading Days", metrics["trading_days"])  

# Health Gauge Chart  
if not signals_df.empty and "hg" in signals_df.columns:  
    st.subheader("🎯 Health Gauge Analysis")  
    fig = go.Figure()  
      
    fig.add_trace(go.Scatter(  
        x=signals_df["date"],  
        y=signals_df["hg"],  
        mode="lines",  
        name="Health Gauge",  
        line=dict(color="#667eea", width=2)  
    ))  
      
    fig.add_hline(y=buy_thresh, line_dash="dash", line_color="#2ed573",   
                 annotation_text=f"Buy Threshold ({buy_thresh})")  
    fig.add_hline(y=sell_thresh, line_dash="dash", line_color="#ff4757",   
                 annotation_text=f"Sell Threshold ({sell_thresh})")  
      
    fig.update_layout(  
        title="Health Gauge Signal Analysis",  
        xaxis_title="Date",  
        yaxis_title="Health Gauge Value",  
        height=400  
    )  
      
    st.plotly_chart(fig, use_container_width=True)  

# Trades table  
if not trades_df.empty:  
    st.subheader("📋 Trade History")  
      
    # Format trades dataframe for display  
    display_trades = trades_df.copy()  
    display_trades["date"] = display_trades["date"].dt.strftime("%Y-%m-%d")  
    display_trades["trade_return_pct"] = display_trades["trade_return_pct"].round(4)  
    display_trades["balance"] = display_trades["balance"].round(2)  
    display_trades["phase_progress"] = display_trades["phase_progress"].round(4)  
      
    st.dataframe(  
        display_trades,  
        use_container_width=True,  
        column_config={  
            "trade_return_pct": st.column_config.NumberColumn(  
                "Return %",  
                format="%.4f"  
            ),  
            "balance": st.column_config.NumberColumn(  
                "Balance",  
                format="$%.2f"  
            ),  
            "phase_progress": st.column_config.NumberColumn(  
                "Phase Progress",  
                format="%.4f"  
            )  
        }  
    )  

# Risk Profile Summary  
with st.expander("📋 Risk Profile Details"):  
    st.json(risk_profile)

if name == "main":
main()