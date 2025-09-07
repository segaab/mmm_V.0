# --- Position Size Calculation with Updated Risk Rules ---
def calculate_position_size(account_balance, risk_percent, leverage, stop_loss_pips, 
                           asset_name, price, margin_alloc_pct=0.4, 
                           maintenance_margin_pct=0.15, min_lot=0.01, lot_step=0.01):
    try:
        asset_class, symbol = asset_classes.get(asset_name, ("FX", "default"))
    except:
        logger.warning(f"Unknown asset: {asset_name}, using FX default")
        asset_class, symbol = "FX", "default"

    # Set maximum effective leverage based on asset class
    if asset_class == "FX":
        max_effective_leverage = 30
    elif asset_class in ["METALS", "OIL", "INDICES"]:
        max_effective_leverage = 10
    else:
        max_effective_leverage = 10

    # Maximum exposure per symbol (20% as per TopOneTrader rule)
    max_total_exposure_pct = 0.20

    # Maximum risk per trade (2% as per prop firm rules)
    risk_percent = min(risk_percent, 0.02)

    contract_size = get_contract_size(asset_class, symbol)
    pip_size = 0.01 if asset_class != "FX" else 0.0001
    position_value_per_lot = contract_size * price
    margin_per_lot = position_value_per_lot / leverage

    # Calculate available margin with updated parameters
    available_margin = account_balance * margin_alloc_pct * (1 - maintenance_margin_pct)
    if available_margin <= 0:
        return 0.0

    # Calculate maximum lots based on various constraints
    max_lots_margin = available_margin / margin_per_lot
    loss_per_lot = contract_size * pip_size * stop_loss_pips
    max_lots_risk = (account_balance * risk_percent) / loss_per_lot if loss_per_lot > 0 else float("inf")

    # Apply maximum exposure limit
    max_value_allowed = account_balance * max_total_exposure_pct
    max_lots_exposure = max_value_allowed / position_value_per_lot

    # Apply maximum effective leverage limit
    max_total_value = account_balance * max_effective_leverage
    max_lots_efflev = max_total_value / position_value_per_lot

    # Take minimum of all constraints
    raw_lots = min(max_lots_margin, max_lots_risk, max_lots_exposure, max_lots_efflev)

    # Round to nearest lot step
    steps = np.floor(raw_lots / lot_step)
    lots = max(0.0, steps * lot_step)
    
    # Apply minimum lot size
    if lots < min_lot:
        lots = 0.0

    return lots

# --- Daily Loss Limit Check ---
def check_daily_loss_limit(current_balance, starting_daily_balance):
    daily_loss_pct = (starting_daily_balance - current_balance) / starting_daily_balance
    return daily_loss_pct <= 0.04  # Using QT Prime's stricter 4% daily limit
