from .base import Signal
from typing import Dict, Any, Optional

VALID_STRATEGY_TYPES = {"scalping", "momentum", "mean_reversion", "breakout"}


def validate_signal(
    signal: Signal,
    market_data: Optional[Dict[str, Any]] = None,
    fee_rate: Optional[float] = None,
    atr: Optional[float] = None,
    price: Optional[float] = None,
    spread_pct: Optional[float] = None,
    timeframe_seconds: Optional[int] = None,
    min_stop_atr_mult: float = 1.0,
    min_hold_time_mult: float = 1.0,
    global_min_risk_reward_ratio: Optional[float] = None,
) -> Signal:
    """
    Validate a trading signal.
    - If action is HOLD, return as-is.
    - Validate strategy_type and required risk parameters.
    - Enforce risk/reward and ATR-based stop rules.
    Confidence is NOT used to reject trades; it will be used later for position sizing.
    """
    if signal.action == "HOLD":
        return signal

    if signal.strategy_type and signal.strategy_type not in VALID_STRATEGY_TYPES:
        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid strategy type: {signal.strategy_type}")

    # Require risk parameters for BUY/SELL
    if signal.action in ("BUY", "SELL"):
        params = signal.strategy_params or {}
        # Determine stop-loss method (default "fixed")
        stop_method = params.get("stop_loss_method", "fixed")
        if stop_method not in ("fixed", "atr_multiple"):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_method")

        if stop_method == "atr_multiple":
            # stop_loss_atr_multiple is required
            if "stop_loss_atr_multiple" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing stop_loss_atr_multiple for atr_multiple method")
            atr_mult = params["stop_loss_atr_multiple"]
            if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_atr_multiple")
            # stop_loss_pct is REQUIRED as a fallback when ATR is unavailable at execution time
            if "stop_loss_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing stop_loss_pct (required as fallback for atr_multiple method)")
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")
        else:  # "fixed"
            if "stop_loss_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: stop_loss_pct")
            sl = params["stop_loss_pct"]
            if not isinstance(sl, (int, float)) or not (0 < sl < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid stop_loss_pct")

        # Enforce minimum fixed stop-loss relative to ATR (if ATR and price are available)
        if stop_method == "fixed" and atr is not None and price is not None and price > 0 and atr > 0:
            atr_pct = atr / price
            min_sl = min_stop_atr_mult * atr_pct
            if sl < min_sl:
                return Signal(
                    action="HOLD",
                    confidence=0.0,
                    reasoning=(
                        f"Fixed stop-loss too tight: must be at least 1.5x ATR "
                        f"(ATR%={atr_pct:.4%}, stop_loss_pct={sl:.4%})"
                    )
                )

        # The rest of the required parameters remain unchanged
        required = ["take_profit_pct", "trailing_stop", "position_size_fraction", "max_hold_time_seconds"]
        for key in required:
            if key not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing required parameter: {key}")
        tp = params["take_profit_pct"]
        if not isinstance(tp, (int, float)) or not (0 < tp < 10.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid take_profit_pct")
        trailing = params["trailing_stop"]
        if not isinstance(trailing, bool):
            return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop must be boolean")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is None or not isinstance(tsd, (int, float)) or not (0 < tsd < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing trailing_stop_distance_pct")
        psf = params["position_size_fraction"]
        if not isinstance(psf, (int, float)) or not (0 < psf <= 1.0):
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid position_size_fraction")
        mht = params["max_hold_time_seconds"]
        if not isinstance(mht, (int, float)) or mht <= 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_hold_time_seconds")
        # Enforce a minimum max hold time relative to the candle timeframe
        if timeframe_seconds is not None and mht < min_hold_time_mult * timeframe_seconds:
            return Signal(
                action="HOLD",
                confidence=0.0,
                reasoning=(
                    f"max_hold_time_seconds ({mht}s) is too short for the "
                    f"timeframe ({timeframe_seconds}s candles); "
                    f"minimum is {min_hold_time_mult * timeframe_seconds}s"
                )
            )

        if "cooldown_after_loss_seconds" not in params:
            return Signal(action="HOLD", confidence=0.0, reasoning="Missing required parameter: cooldown_after_loss_seconds")
        cd = params["cooldown_after_loss_seconds"]
        if not isinstance(cd, (int, float)) or cd < 0:
            return Signal(action="HOLD", confidence=0.0, reasoning="Invalid cooldown_after_loss_seconds")

        # Optional new parameters
        if "trailing_stop_activation_pct" in params:
            tsa = params["trailing_stop_activation_pct"]
            if not isinstance(tsa, (int, float)) or not (0 <= tsa <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid trailing_stop_activation_pct")
        if "trailing_take_profit" in params:
            ttp = params["trailing_take_profit"]
            if not isinstance(ttp, bool):
                return Signal(action="HOLD", confidence=0.0, reasoning="trailing_take_profit must be boolean")
            if ttp:
                ttp_dist = params.get("trailing_take_profit_distance_pct")
                if ttp_dist is None or not isinstance(ttp_dist, (int, float)) or not (0 < ttp_dist < 1.0):
                    return Signal(action="HOLD", confidence=0.0, reasoning="Invalid or missing trailing_take_profit_distance_pct")
        if "breakeven_activation_pct" in params:
            bap = params["breakeven_activation_pct"]
            if not isinstance(bap, (int, float)) or not (0 < bap <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid breakeven_activation_pct")
        if "lock_profit_activation_pct" in params:
            lpa = params["lock_profit_activation_pct"]
            if not isinstance(lpa, (int, float)) or not (0 < lpa <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid lock_profit_activation_pct")
            if "lock_profit_level_pct" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing lock_profit_level_pct")
            lpl = params["lock_profit_level_pct"]
            if not isinstance(lpl, (int, float)) or not (0 < lpl < lpa):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid lock_profit_level_pct (must be < activation)")
        if "partial_take_profit_pct" in params:
            ptp = params["partial_take_profit_pct"]
            if not isinstance(ptp, (int, float)) or not (0 < ptp <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_pct")
            if "partial_take_profit_fraction" not in params:
                return Signal(action="HOLD", confidence=0.0, reasoning="Missing partial_take_profit_fraction")
            ptf = params["partial_take_profit_fraction"]
            if not isinstance(ptf, (int, float)) or not (0 < ptf <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid partial_take_profit_fraction")
            # Partial TP must be less than main TP
            if ptp >= tp:
                return Signal(action="HOLD", confidence=0.0, reasoning="partial_take_profit_pct must be less than take_profit_pct")
        # --- Multiple partial take-profit levels ---
        if "partial_take_profit_levels" in params:
            levels = params["partial_take_profit_levels"]
            if not isinstance(levels, list) or len(levels) == 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="partial_take_profit_levels must be a non-empty array")
            total_fraction = 0.0
            prev_pct = 0.0
            for i, level in enumerate(levels):
                if not isinstance(level, dict):
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid partial_take_profit_levels[{i}]")
                lvl_pct = level.get("take_profit_pct")
                lvl_frac = level.get("fraction")
                if not isinstance(lvl_pct, (int, float)) or not (0 < lvl_pct <= 1.0):
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid take_profit_pct in level {i}")
                if not isinstance(lvl_frac, (int, float)) or not (0 < lvl_frac <= 1.0):
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid fraction in level {i}")
                min_depth = level.get("min_depth")
                if min_depth is not None:
                    if not isinstance(min_depth, (int, float)) or min_depth <= 0:
                        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid min_depth in level {i}")
                max_time = level.get("max_time_seconds")
                if max_time is not None:
                    if not isinstance(max_time, (int, float)) or max_time <= 0:
                        return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid max_time_seconds in level {i}")
                if lvl_pct <= prev_pct:
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Levels must be in increasing take_profit_pct order")
                if lvl_pct >= tp:
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Level {i} take_profit_pct must be less than main take_profit_pct")
                total_fraction += lvl_frac
                prev_pct = lvl_pct
            if total_fraction > 1.0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Sum of partial take-profit fractions exceeds 1.0")
        if "max_risk_per_trade_pct" in params:
            mrp = params["max_risk_per_trade_pct"]
            if not isinstance(mrp, (int, float)) or not (0 < mrp <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_risk_per_trade_pct")
        if "min_profit_per_trade" in params:
            mpp = params["min_profit_per_trade"]
            if not isinstance(mpp, (int, float)) or mpp < 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_profit_per_trade")
        mrr = params.get("min_risk_reward_ratio")
        if mrr is None and global_min_risk_reward_ratio is not None:
            mrr = global_min_risk_reward_ratio
        if mrr is not None:
            if not isinstance(mrr, (int, float)) or mrr <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_risk_reward_ratio")
            # Enforce the ratio if both sl and tp are available
            if sl is not None and tp is not None:
                if tp / sl < mrr:
                    return Signal(
                        action="HOLD",
                        confidence=0.0,
                        reasoning=f"Risk/reward ratio {tp/sl:.2f} is below minimum {mrr:.2f}"
                    )
        if "max_spread_pct" in params:
            msp = params["max_spread_pct"]
            if not isinstance(msp, (int, float)) or msp <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_spread_pct")
        if "min_depth_at_take_profit" in params:
            mdatp = params["min_depth_at_take_profit"]
            if not isinstance(mdatp, (int, float)) or mdatp <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_depth_at_take_profit")
        if "max_slippage_pct" in params:
            msp = params["max_slippage_pct"]
            if not isinstance(msp, (int, float)) or msp <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_slippage_pct")
        if "max_unrealized_loss_pct" in params:
            mul = params["max_unrealized_loss_pct"]
            if not isinstance(mul, (int, float)) or not (0 < mul < 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid max_unrealized_loss_pct")
            if sl is not None and mul >= sl:
                return Signal(action="HOLD", confidence=0.0, reasoning="max_unrealized_loss_pct must be less than stop_loss_pct")
        if "min_confidence" in params:
            mc = params["min_confidence"]
            if not isinstance(mc, (int, float)) or not (0.0 <= mc <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid min_confidence")
        if "news_sentiment_exit_threshold" in params:
            nst = params["news_sentiment_exit_threshold"]
            if not isinstance(nst, (int, float)) or not (-1.0 <= nst <= 1.0):
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid news_sentiment_exit_threshold")
        if "strategy_interval_seconds" in params:
            si = params["strategy_interval_seconds"]
            if not isinstance(si, (int, float)) or si <= 0:
                return Signal(action="HOLD", confidence=0.0, reasoning="Invalid strategy_interval_seconds")

        # --- Native order type validation ---
        order_type = params.get("order_type")
        if order_type is not None:
            if order_type not in ("market", "limit", "stop", "stop_limit", "trailing_stop"):
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid order_type: {order_type}")
            if order_type in ("stop", "stop_limit"):
                sp = params.get("stop_price")
                if sp is None or not isinstance(sp, (int, float)) or sp <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing or invalid stop_price for order_type={order_type}")
            if order_type in ("limit", "stop_limit"):
                lp = params.get("limit_price")
                if lp is None or not isinstance(lp, (int, float)) or lp <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning=f"Missing or invalid limit_price for order_type={order_type}")
            if order_type == "trailing_stop":
                to = params.get("trail_offset")
                if to is None or not isinstance(to, (int, float)) or to <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning="Missing or invalid trail_offset for order_type=trailing_stop")

        # --- Exit order type validation ---
        stop_loss_ot = params.get("stop_loss_order_type")
        if stop_loss_ot is not None:
            if stop_loss_ot not in ("market", "stop", "stop_limit", "trailing_stop"):
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid stop_loss_order_type: {stop_loss_ot}")
            if stop_loss_ot in ("stop", "stop_limit"):
                sp = params.get("stop_loss_stop_price")
                if sp is None or not isinstance(sp, (int, float)) or sp <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning="Missing or invalid stop_loss_stop_price for stop_loss_order_type")
            if stop_loss_ot == "stop_limit":
                lp = params.get("stop_loss_limit_price")
                if lp is None or not isinstance(lp, (int, float)) or lp <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning="Missing or invalid stop_loss_limit_price for stop_loss_order_type=stop_limit")
            if stop_loss_ot == "trailing_stop":
                to = params.get("stop_loss_trail_offset")
                if to is None or not isinstance(to, (int, float)) or to <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning="Missing or invalid stop_loss_trail_offset for stop_loss_order_type=trailing_stop")

        take_profit_ot = params.get("take_profit_order_type")
        if take_profit_ot is not None:
            if take_profit_ot not in ("limit", "market"):
                return Signal(action="HOLD", confidence=0.0, reasoning=f"Invalid take_profit_order_type: {take_profit_ot}")
            if take_profit_ot == "limit":
                lp = params.get("take_profit_limit_price")
                if lp is None or not isinstance(lp, (int, float)) or lp <= 0:
                    return Signal(action="HOLD", confidence=0.0, reasoning="Missing or invalid take_profit_limit_price for take_profit_order_type=limit")

        # Logical consistency checks (no hardcoded values)
        if sl is not None and tp <= sl:
            return Signal(action="HOLD", confidence=0.0, reasoning="take_profit_pct must be greater than stop_loss_pct")
        if trailing:
            tsd = params.get("trailing_stop_distance_pct")
            if tsd is not None and sl is not None and tsd >= sl:
                return Signal(action="HOLD", confidence=0.0, reasoning="trailing_stop_distance_pct must be less than stop_loss_pct")

    return signal
