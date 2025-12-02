"""
Price Action Volumetric Order Blocks -> signal & visualization (matplotlib)

This script implements a close analogue of the Pine Script logic you posted:
- symmetric pivot detection (ta.pivothigh / ta.pivotlow style)
- when price breaks a pivot (BOS/MSB), select a candidate candle inside the
  previous `swing_length` bars to be the order block (highest green or lowest red)
- compute bullish/bearish strength by summing volume over a window ending at
  the selected candle (same behavior as the Pine script)
- create OBs, optionally hide overlaps, track OBs until they are violated,
  and generate buy/sell signals at the next bar after OB creation when the
  strength ratio exceeds a threshold

How to use:
- Adjust parameters in the PARAMETERS section.
- Provide your OHLCV data as a list-of-dicts (see example below) or load into
  a pandas DataFrame with columns: ['time'(ms),'open','high','low','close','volume'].
- Run the script. It will plot a line-chart of close prices, shaded OB rectangles,
  and buy/sell markers. It also returns `signals` and `obs` from detect_order_blocks().

Dependencies:
- pandas, numpy, matplotlib

Author: Copied-style implementation from Pine Script logic (converted to Python)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timezone


# =========================
# Data structures
# =========================
@dataclass
class OrderBlock:
    kind: str            # "bullish" or "bearish"
    top: float
    btm: float
    start_idx: int       # index of the selected candle (barStart in pine)
    create_idx: int      # index where OB was created (breakout bar)
    create_time: pd.Timestamp
    violated_idx: Optional[int] = None
    violated_time: Optional[pd.Timestamp] = None
    bullish_str: float = 0.0
    bearish_str: float = 0.0
    vol: float = 0.0      # volume of selected candle

@dataclass
class Signal:
    idx: int
    time: pd.Timestamp
    kind: str    # "buy" or "sell"
    price: float
    ob_idx: Optional[int] = None  # index of OB in returned list (None if OB was skipped)

@dataclass
class Position:
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    position_type: str  # "long" or "short"
    stop_loss: float
    take_profit: float
    ob_size: float
    sl_distance: float
    tp_distance: float
    position_size: float = 0.0         # number of units (e.g., BTC)
    capital_at_risk: float = 0.0       # capital risked on this trade
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "SL", "TP", "trailing_stop", or "signal"
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_dollars: Optional[float] = None  # PnL in dollar terms based on position size
    initial_stop_loss: Optional[float] = None  # original SL before trailing
    highest_price: Optional[float] = None  # for trailing stop (long positions)
    lowest_price: Optional[float] = None   # for trailing stop (short positions)
    trailing_active: bool = False       # whether trailing stop is active
    trailing_activation_idx: Optional[int] = None  # candle index when trailing was activated
    last_trailing_update_price: Optional[float] = None  # last price where trailing stop was updated
    # track fees per trade
    total_cost_per_unit: Optional[float] = None   # sum of entry+exit fees per unit (same units as price)
    total_fees_dollars: Optional[float] = None    # total fees paid for this trade (cost_per_unit * position_size)
    # holding-period counter: number of bars since the last reinforcing signal (or since entry)
    holding_counter: int = 0

# =========================
# Helper functions
# =========================
def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """
    Return a Series with pivot highs placed at the confirmation index (i + right).
    Mirrors Pine's ta.pivothigh(left,right) semantics.
    """
    n = len(high)
    piv = pd.Series(np.nan, index=high.index)
    for i in range(left, n - right):
        center = high.iat[i]
        # use .iloc for slices; .iat only for scalar access
        left_max = high.iloc[i-left:i].max() if left > 0 else -np.inf
        right_max = high.iloc[i+1:i+1+right].max() if right > 0 else -np.inf
        if center > left_max and center > right_max:
            piv.iat[i + right] = center
    return piv

def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    n = len(low)
    piv = pd.Series(np.nan, index=low.index)
    for i in range(left, n - right):
        center = low.iat[i]
        left_min = low.iloc[i-left:i].min() if left > 0 else np.inf
        right_min = low.iloc[i+1:i+1+right].min() if right > 0 else np.inf
        if center < left_min and center < right_min:
            piv.iat[i + right] = center
    return piv

def is_price_overlap(new_top: float, new_btm: float, existing: OrderBlock) -> bool:
    # overlap if intervals intersect
    return not (new_top < existing.btm or new_btm > existing.top)

def calculate_strengths(df: pd.DataFrame, selected_idx: int, bars_to_consider: int) -> Tuple[float, float]:
    """
    Sum volume backwards from selected_idx for bars_to_consider bars.
    A bar is 'bearish' if open > close; otherwise bullish (matches Pine script logic).
    """
    bullish = 0.0
    bearish = 0.0
    start = max(0, selected_idx - (bars_to_consider - 1))
    for i in range(selected_idx, start - 1, -1):
        vol = float(df['volume'].iat[i])
        if df['open'].iat[i] > df['close'].iat[i]:
            bearish += vol
        else:
            bullish += vol
    return bullish, bearish


# Helper to finalize exits and compute fees/pnl
def finalize_position_exit(pos: Position,
                           exit_idx: int,
                           exit_time_ms,
                           exit_price: float,
                           exit_reason: str,
                           commission_percent: float,
                           slippage_percent: float) -> float:
    """
    Populate pos exit fields, compute pnl, compute total fees per unit and total fees dollars,
    compute pnl_dollars (net of fees) and return pnl_dollars.
    NOTE: keeps the existing fee model: price * ((commission+slippage)/100) per side.
    """
    pos.exit_idx = exit_idx
    pos.exit_time = exit_time_ms
    pos.exit_price = exit_price
    pos.exit_reason = exit_reason

    if pos.position_type == "long":
        pos.pnl = pos.exit_price - pos.entry_price
    else:
        pos.pnl = pos.entry_price - pos.exit_price

    pos.pnl_percent = (pos.pnl / pos.entry_price) * 100 if pos.entry_price != 0 else None

    total_cost_per_unit = pos.entry_price * ((commission_percent + slippage_percent) / 100)
    total_cost_per_unit += pos.exit_price * ((commission_percent + slippage_percent) / 100)
    pos.total_cost_per_unit = total_cost_per_unit
    pos.total_fees_dollars = total_cost_per_unit * pos.position_size

    pos.pnl_dollars = (pos.pnl * pos.position_size) - pos.total_fees_dollars
    return pos.pnl_dollars

# =========================
# Core detection function
# =========================
def detect_order_blocks(df: pd.DataFrame,
                        swing_length: int = 8,
                        hide_overlap: bool = True,
                        show_last_x_ob: int = 4,
                        violation_type: str = "Wick",
                        min_strength_ratio: float = 0.6,
                        stop_loss_multiplier: float = 1.1,
                        take_profit_multiplier: float = 3.0,
                        max_concurrent_positions: int = 1,
                        initial_capital: float = 10000.0,
                        risk_per_trade_percent: float = 2.0,
                        trailing_stop_activation: float = 1.25,
                        trailing_stop_percent: float = 2.0,
                        trailing_stop_buffer_candles: int = 3,
                        trailing_buffer_blocks_updates: bool = True,
                        trailing_move_on_activation: bool = True,
                        trailing_stop_update_threshold: float = 1.0,
                        trailing_atr_mult: float = 1.25,
                        use_fixed_capital: bool = False,
                        max_position_size_usd: float = 100000.0,
                        entry_price_mode: str = "close",
                        commission_percent: float = 0.1,
                        slippage_percent: float = 0.05,
                        # New tuning params
                        break_atr_mult: float = 0.0,
                        ob_search_window: Optional[int] = None,
                        ob_min_sl_atr_mult: float = 0.5,
                        ob_min_sl_pct: float = 0.001,
                        # ATR tuning: explicit period or multiplier of swing_length
                        atr_period: Optional[int] = None,
                        atr_period_mult: Optional[float] = None,
                        holding_period_bars: int = 0, reverse_signals: bool = False) -> Tuple[List[OrderBlock], List[Signal], List[Position], pd.DataFrame]:
    """
    Walk forward through data, emulate Pine script behavior:
    - compute pivots (confirmed at i)
    - when a breakout close beyond the pivot occurs and pivot hasn't been crossed, create OB
    - track OB until violated; generate signal at the next bar after OB creation if strength ratio condition met
    - manage positions with stop loss and take profit based on order block size
    - position sizing based on risk per trade percentage
    - trailing stop loss to lock in profits (percentage-based with buffer and threshold)
    - configurable entry price (close or worst price in candle)
    - includes commission and slippage costs
    
    Trailing Stop Parameters:
    - trailing_stop_percent: percentage below highest (longs) or above lowest (shorts) to trail
    - trailing_stop_buffer_candles: number of candles to wait after activation before allowing exit
    - trailing_stop_update_threshold: minimum price movement % required to update trailing stop
    """
    n = len(df)
    highs = df['high']
    lows = df['low']
    closes = df['close']
    # Determine effective ATR period (backwards-compatible default using atr_period_mult)
    # Resolve effective atr_period_mult: prefer explicit function arg, then module-level config, then default 0.5
    try:
        if atr_period_mult is None:
            atr_mult_use = float(globals().get('atr_period_mult', 0.5))
        else:
            atr_mult_use = float(atr_period_mult)
    except Exception:
        atr_mult_use = 0.5

    try:
        if atr_period is None:
            # compute from swing_length * atr_mult_use, fall back to 14 if computation fails
            atr_period_eff = int(round(float(swing_length) * float(atr_mult_use)))
        else:
            atr_period_eff = int(atr_period)
    except Exception:
        atr_period_eff = 14
    atr_period_eff = max(1, atr_period_eff)

    # Compute several candidate range/volatility series and pick according to atr_method
    # 1) legacy per-candle high-low SMA (keeps previous default behavior)
    hl_sma = (highs - lows).rolling(window=atr_period_eff, min_periods=1).mean().bfill()

    # 2) structural high-low range over window (captures swing amplitude including wicks)
    max_high = highs.rolling(window=atr_period_eff, min_periods=1).max().bfill()
    min_low = lows.rolling(window=atr_period_eff, min_periods=1).min().bfill()
    maxmin_range = (max_high - min_low).bfill()

    # 3) close-based range (highest close - lowest close over window) — ignores intrabar wicks
    max_close = closes.rolling(window=atr_period_eff, min_periods=1).max().bfill()
    min_close = closes.rolling(window=atr_period_eff, min_periods=1).min().bfill()
    close_range = (max_close - min_close).bfill()

    # select series based on method preference: prefer function arg if provided, else module-level setting
    try:
        method = globals().get('atr_method', 'highlow_sma')
        if 'atr_method' in locals() and atr_method is not None:
            method = atr_method
    except Exception:
        method = globals().get('atr_method', 'highlow_sma')

    if method == 'maxmin_range':
        atr = maxmin_range * float(globals().get('range_percent', 1.0))
    elif method == 'close_range':
        atr = close_range * float(globals().get('range_percent', 1.0))
    else:
        atr = hl_sma

    piv_h = pivot_high(highs, swing_length, swing_length)
    piv_l = pivot_low(lows, swing_length, swing_length)

    # EMA series for optional entry filtering
    try:
        ema_period_use = int(globals().get('ema_period', 50))
    except Exception:
        ema_period_use = 50
    try:
        ema = closes.ewm(span=max(1, ema_period_use), adjust=False).mean()
    except Exception:
        ema = pd.Series([np.nan] * n, index=df.index)

    # Keep track of last active pivot objects (store pivot price and crossed flag)
    last_top = None      # (value, pivot_index, crossed_flag)
    last_btm = None

    obs: List[OrderBlock] = []
    signals: List[Signal] = []
    positions: List[Position] = []
    open_positions: List[Position] = []
    current_capital = initial_capital  # Track running capital
    equity_ts = []

    # For quick lookup of OB indices to mark violations
    active_obs_indices: List[int] = []

    # Trailing counters for diagnostics
    n_trailing_activations = 0
    n_trailing_updates = 0
    n_trailing_updates_skipped = 0
    n_trailing_exits = 0

    for t in range(n):
        # First, update trailing stops and check for exits
        still_open = []
        for pos in open_positions:
            closed = False
            current_high = float(df['high'].iat[t])
            current_low = float(df['low'].iat[t])
            
            if pos.position_type == "long":
                # Update highest price seen for trailing stop
                if pos.highest_price is None:
                    pos.highest_price = pos.entry_price
                pos.highest_price = max(pos.highest_price, current_high)
                
                # Check if we should activate trailing stop
                unrealized_profit = pos.highest_price - pos.entry_price
                if not pos.trailing_active and unrealized_profit >= (pos.sl_distance * trailing_stop_activation):
                    pos.trailing_active = True
                    pos.initial_stop_loss = pos.stop_loss
                    pos.trailing_activation_idx = t
                    pos.last_trailing_update_price = pos.highest_price
                    # Activation: compute ATR-aware trailing level and optionally move SL immediately
                    try:
                        n_trailing_activations += 1
                    except Exception:
                        pass
                    try:
                        atr_here = float(atr.iat[t]) if (not np.isnan(atr.iat[t])) else 0.0
                    except Exception:
                        atr_here = 0.0
                    try:
                        pct_level = pos.highest_price * (1 - trailing_stop_percent / 100)
                        atr_level = pos.highest_price - (trailing_atr_mult * atr_here)
                        new_trailing_sl = max(pct_level, atr_level)
                    except Exception:
                        new_trailing_sl = pos.highest_price * (1 - trailing_stop_percent / 100)

                    prev_sl = pos.stop_loss
                    if trailing_move_on_activation:
                        pos.stop_loss = max(pos.stop_loss, new_trailing_sl)
                    # Diagnostic: trailing activation (show initial and updated SL)
                    try:
                        print(f"[TRAILING ACTIVATE] LONG entry_idx={pos.entry_idx} t={t} entry={pos.entry_price:.6f} highest={pos.highest_price:.6f} unrealized_profit={unrealized_profit:.6f} sl_distance={pos.sl_distance:.6f} activation_thresh={pos.sl_distance * trailing_stop_activation:.6f} prev_sl={prev_sl:.6f} updated_sl={pos.stop_loss:.6f} atr_here={atr_here:.6f} atr_mult={trailing_atr_mult}")
                    except Exception:
                        pass
                
                # Update trailing stop if active AND buffer period has passed
                if pos.trailing_active:
                    # Check if buffer period has passed
                    candles_since_activation = t - pos.trailing_activation_idx
                    # Compute ATR-aware trailing level for potential update
                    try:
                        atr_here = float(atr.iat[t]) if (not np.isnan(atr.iat[t])) else 0.0
                    except Exception:
                        atr_here = 0.0
                    try:
                        pct_level = pos.highest_price * (1 - trailing_stop_percent / 100)
                        atr_level = pos.highest_price - (trailing_atr_mult * atr_here)
                        new_trailing_sl = max(pct_level, atr_level)
                    except Exception:
                        new_trailing_sl = pos.highest_price * (1 - trailing_stop_percent / 100)

                    # If updates are blocked by buffer, skip update until buffer elapses
                    if trailing_buffer_blocks_updates and candles_since_activation < trailing_stop_buffer_candles:
                        n_trailing_updates_skipped += 1
                    else:
                        # Only update if price moved enough since last update (threshold) and SL would move up
                        if pos.last_trailing_update_price is not None:
                            price_move_percent = ((pos.highest_price - pos.last_trailing_update_price) / pos.last_trailing_update_price) * 100
                            if price_move_percent >= trailing_stop_update_threshold:
                                prev_sl = pos.stop_loss
                                pos.stop_loss = max(pos.stop_loss, new_trailing_sl)
                                pos.last_trailing_update_price = pos.highest_price
                                n_trailing_updates += 1
                                try:
                                    print(f"[TRAILING UPDATE] LONG entry_idx={pos.entry_idx} t={t} prev_sl={prev_sl:.6f} new_trailing_sl={new_trailing_sl:.6f} updated_sl={pos.stop_loss:.6f} price_move_pct={price_move_percent:.3f} atr_here={atr_here:.6f}")
                                except Exception:
                                    pass
                        else:
                            prev_sl = pos.initial_stop_loss if pos.initial_stop_loss is not None else pos.stop_loss
                            pos.stop_loss = max(pos.stop_loss, new_trailing_sl)
                            pos.last_trailing_update_price = pos.highest_price
                            n_trailing_updates += 1
                            try:
                                print(f"[TRAILING UPDATE] LONG entry_idx={pos.entry_idx} t={t} prev_sl={prev_sl:.6f} new_trailing_sl={new_trailing_sl:.6f} updated_sl={pos.stop_loss:.6f} (first_update) atr_here={atr_here:.6f}")
                            except Exception:
                                pass

                    # Check if SL hit, but only after buffer period
                    if candles_since_activation >= trailing_stop_buffer_candles and current_low <= pos.stop_loss:
                        try:
                            print(f"[TRAILING EXIT] LONG entry_idx={pos.entry_idx} t={t} current_low={current_low:.6f} stop_loss={pos.stop_loss:.6f} candles_since_activation={candles_since_activation}")
                        except Exception:
                            pass
                        n_trailing_exits += 1
                        pnl_dollars = finalize_position_exit(
                            pos,
                            exit_idx=t,
                            exit_time_ms=df['timestamp'].iat[t],
                            exit_price=pos.stop_loss,
                            exit_reason="trailing_stop",
                            commission_percent=commission_percent,
                            slippage_percent=slippage_percent
                        )
                        current_capital += pnl_dollars
                        closed = True
                
                # Check if initial SL or TP hit (these work regardless of trailing stop status)
                if not closed and current_low <= pos.stop_loss and not pos.trailing_active:
                    # Initial stop loss hit (before trailing activated)
                    pnl_dollars = finalize_position_exit(
                        pos,
                        exit_idx=t,
                        exit_time_ms=df['timestamp'].iat[t],
                        exit_price=pos.stop_loss,
                        exit_reason="SL",
                        commission_percent=commission_percent,
                        slippage_percent=slippage_percent
                    )
                    current_capital += pnl_dollars
                    closed = True
                elif not closed and current_high >= pos.take_profit:
                    # Take profit hit
                    pnl_dollars = finalize_position_exit(
                        pos,
                        exit_idx=t,
                        exit_time_ms=df['timestamp'].iat[t],
                        exit_price=pos.take_profit,
                        exit_reason="TP",
                        commission_percent=commission_percent,
                        slippage_percent=slippage_percent
                    )
                    current_capital += pnl_dollars
                    closed = True
                    
            elif pos.position_type == "short":
                # Update lowest price seen for trailing stop
                if pos.lowest_price is None:
                    pos.lowest_price = pos.entry_price
                pos.lowest_price = min(pos.lowest_price, current_low)
                
                # Check if we should activate trailing stop
                unrealized_profit = pos.entry_price - pos.lowest_price
                if not pos.trailing_active and unrealized_profit >= (pos.sl_distance * trailing_stop_activation):
                    pos.trailing_active = True
                    pos.initial_stop_loss = pos.stop_loss
                    pos.trailing_activation_idx = t
                    pos.last_trailing_update_price = pos.lowest_price
                    # Activation: compute ATR-aware trailing level and optionally move SL immediately
                    try:
                        n_trailing_activations += 1
                    except Exception:
                        pass
                    try:
                        atr_here = float(atr.iat[t]) if (not np.isnan(atr.iat[t])) else 0.0
                    except Exception:
                        atr_here = 0.0
                    try:
                        pct_level = pos.lowest_price * (1 + trailing_stop_percent / 100)
                        atr_level = pos.lowest_price + (trailing_atr_mult * atr_here)
                        new_trailing_sl = min(pct_level, atr_level)
                    except Exception:
                        new_trailing_sl = pos.lowest_price * (1 + trailing_stop_percent / 100)

                    prev_sl = pos.stop_loss
                    if trailing_move_on_activation:
                        pos.stop_loss = min(pos.stop_loss, new_trailing_sl)
                    try:
                        print(f"[TRAILING ACTIVATE] SHORT entry_idx={pos.entry_idx} t={t} entry={pos.entry_price:.6f} lowest={pos.lowest_price:.6f} unrealized_profit={unrealized_profit:.6f} sl_distance={pos.sl_distance:.6f} activation_thresh={pos.sl_distance * trailing_stop_activation:.6f} prev_sl={prev_sl:.6f} updated_sl={pos.stop_loss:.6f} atr_here={atr_here:.6f} atr_mult={trailing_atr_mult}")
                    except Exception:
                        pass
                
                # Update trailing stop if active AND buffer period has passed
                if pos.trailing_active:
                    # Check if buffer period has passed
                    candles_since_activation = t - pos.trailing_activation_idx
                    # Compute ATR-aware trailing level for potential update
                    try:
                        atr_here = float(atr.iat[t]) if (not np.isnan(atr.iat[t])) else 0.0
                    except Exception:
                        atr_here = 0.0
                    try:
                        pct_level = pos.lowest_price * (1 + trailing_stop_percent / 100)
                        atr_level = pos.lowest_price + (trailing_atr_mult * atr_here)
                        new_trailing_sl = min(pct_level, atr_level)
                    except Exception:
                        new_trailing_sl = pos.lowest_price * (1 + trailing_stop_percent / 100)

                    # If updates are blocked by buffer, skip update until buffer elapses
                    if trailing_buffer_blocks_updates and candles_since_activation < trailing_stop_buffer_candles:
                        n_trailing_updates_skipped += 1
                    else:
                        if pos.last_trailing_update_price is not None:
                            price_move_percent = ((pos.last_trailing_update_price - pos.lowest_price) / pos.last_trailing_update_price) * 100
                            if price_move_percent >= trailing_stop_update_threshold:
                                prev_sl = pos.stop_loss
                                pos.stop_loss = min(pos.stop_loss, new_trailing_sl)
                                pos.last_trailing_update_price = pos.lowest_price
                                n_trailing_updates += 1
                                try:
                                    print(f"[TRAILING UPDATE] SHORT entry_idx={pos.entry_idx} t={t} prev_sl={prev_sl:.6f} new_trailing_sl={new_trailing_sl:.6f} updated_sl={pos.stop_loss:.6f} price_move_pct={price_move_percent:.3f} atr_here={atr_here:.6f}")
                                except Exception:
                                    pass
                        else:
                            prev_sl = pos.initial_stop_loss if pos.initial_stop_loss is not None else pos.stop_loss
                            pos.stop_loss = min(pos.stop_loss, new_trailing_sl)
                            pos.last_trailing_update_price = pos.lowest_price
                            n_trailing_updates += 1
                            try:
                                print(f"[TRAILING UPDATE] SHORT entry_idx={pos.entry_idx} t={t} prev_sl={prev_sl:.6f} new_trailing_sl={new_trailing_sl:.6f} updated_sl={pos.stop_loss:.6f} (first_update) atr_here={atr_here:.6f}")
                            except Exception:
                                pass

                    # Check if SL hit, but only after buffer period
                    if candles_since_activation >= trailing_stop_buffer_candles and current_high >= pos.stop_loss:
                        try:
                            print(f"[TRAILING EXIT] SHORT entry_idx={pos.entry_idx} t={t} current_high={current_high:.6f} stop_loss={pos.stop_loss:.6f} candles_since_activation={candles_since_activation}")
                        except Exception:
                            pass
                        n_trailing_exits += 1
                        pnl_dollars = finalize_position_exit(
                            pos,
                            exit_idx=t,
                            exit_time_ms=df['timestamp'].iat[t],
                            exit_price=pos.stop_loss,
                            exit_reason="trailing_stop",
                            commission_percent=commission_percent,
                            slippage_percent=slippage_percent
                        )
                        current_capital += pnl_dollars
                        closed = True
                
                # Check if initial SL or TP hit (these work regardless of trailing stop status)
                if not closed and current_high >= pos.stop_loss and not pos.trailing_active:
                    # Initial stop loss hit (before trailing activated)
                    pnl_dollars = finalize_position_exit(
                        pos,
                        exit_idx=t,
                        exit_time_ms=df['timestamp'].iat[t],
                        exit_price=pos.stop_loss,
                        exit_reason="SL",
                        commission_percent=commission_percent,
                        slippage_percent=slippage_percent
                    )
                    current_capital += pnl_dollars
                    closed = True
                elif not closed and current_low <= pos.take_profit:
                    # Take profit hit
                    pnl_dollars = finalize_position_exit(
                        pos,
                        exit_idx=t,
                        exit_time_ms=df['timestamp'].iat[t],
                        exit_price=pos.take_profit,
                        exit_reason="TP",
                        commission_percent=commission_percent,
                        slippage_percent=slippage_percent
                    )
                    current_capital += pnl_dollars
                    closed = True
            
            if not closed:
                # Increment holding counter (number of bars since last same-side signal)
                try:
                    # only increment if holding_period is enabled
                    if holding_period_bars and holding_period_bars > 0:
                        pos.holding_counter = (pos.holding_counter or 0) + 1
                        # if reached threshold, close at market (use close price)
                        if pos.holding_counter >= holding_period_bars:
                            try:
                                print(f"[HOLDING EXIT] {pos.position_type.upper()} entry_idx={pos.entry_idx} t={t} holding_counter={pos.holding_counter} threshold={holding_period_bars}")
                            except Exception:
                                pass
                            pnl_dollars = finalize_position_exit(
                                pos,
                                exit_idx=t,
                                exit_time_ms=df['timestamp'].iat[t],
                                exit_price=float(df['close'].iat[t]),
                                exit_reason="holding_period",
                                commission_percent=commission_percent,
                                slippage_percent=slippage_percent
                            )
                            current_capital += pnl_dollars
                            closed = True
                except Exception:
                    pass

                if not closed:
                    still_open.append(pos)
        
        open_positions = still_open

        # record equity at this bar (mark-to-market using close price)
        try:
            mark_price = float(df['close'].iat[t])
        except Exception:
            mark_price = None
        unrealized = 0.0
        if mark_price is not None:
            for pos in open_positions:
                if pos.position_size is None:
                    continue
                if pos.position_type == 'long':
                    unrealized += (mark_price - pos.entry_price) * pos.position_size
                else:
                    unrealized += (pos.entry_price - mark_price) * pos.position_size
        equity = current_capital + unrealized
        equity_ts.append({'time': int(df['timestamp'].iat[t]), 'equity': float(equity)})
        
        # if a pivot high is confirmed at t
        if not np.isnan(piv_h.iat[t]):
            # pivot bar is at t - swing_length (center), price = highs[t - swing_length]
            pivot_price = piv_h.iat[t]
            last_top = {'price': pivot_price, 'idx': t, 'crossed': False}

        if not np.isnan(piv_l.iat[t]):
            pivot_price = piv_l.iat[t]
            last_btm = {'price': pivot_price, 'idx': t, 'crossed': False}

        # Check for break below last_btm (bearish OB creation)
        if last_btm is not None and (not last_btm['crossed']):
            # breakout must clear pivot by ATR*mult if break_atr_mult > 0
            pivot_price = float(last_btm['price'])
            atr_val = float(atr.iat[t]) if not np.isnan(atr.iat[t]) else 0.0
            required_break = pivot_price - (break_atr_mult * atr_val)
            if float(df['close'].iat[t]) < required_break:
                # create bearish OB: find highest green candle in previous swing_length bars (t-1 .. t-swing_length)
                last_btm['crossed'] = True
                # search previous swing_length bars for close > open and max high
                best_top = -np.inf
                best_btm = np.nan
                best_idx = None
                best_vol = np.nan
                for i in range(1, swing_length + 1):
                    idx = t - i
                    if idx < 0:
                        break
                    if df['close'].iat[idx] > df['open'].iat[idx]:
                        if df['high'].iat[idx] > best_top:
                            best_top = df['high'].iat[idx]
                            best_btm = df['low'].iat[idx]
                            best_idx = idx
                            best_vol = df['volume'].iat[idx]
                if best_idx is None:
                    # fallback: choose the maximum high in the window (regardless of color)
                    for i in range(1, swing_length + 1):
                        idx = t - i
                        if idx < 0:
                            break
                        if df['high'].iat[idx] > best_top:
                            best_top = df['high'].iat[idx]
                            best_btm = df['low'].iat[idx]
                            best_idx = idx
                            best_vol = df['volume'].iat[idx]

                if best_idx is not None:
                    # determine search window for OB geometry
                    # ensure search_w is an integer (user may pass a float like swing_length*0.25)
                    search_w = int(ob_search_window) if (ob_search_window is not None) else int(swing_length)
                    search_w = max(1, search_w)
                    half = search_w // 2
                    # ensure positional indices for iloc are integers
                    ws = max(0, int(best_idx) - int(half))
                    we = min(n - 1, int(best_idx) + int(half))
                    ob_top = float(highs.iloc[ws:we+1].max())
                    ob_btm = float(lows.iloc[ws:we+1].min())
                    window_vol = float(df['volume'].iloc[ws:we+1].sum())

                    bullish, bearish = calculate_strengths(df, best_idx, swing_length)
                    total = bullish + bearish
                    if total < min_total_volume:
                        total = min_total_volume
                    ob_candidate = OrderBlock(
                        kind='bearish',
                        top=ob_top,
                        btm=ob_btm,
                        start_idx=best_idx,
                        create_idx=t,
                        create_time=df['timestamp'].iat[t],
                        bullish_str=float(bullish),
                        bearish_str=float(bearish),
                        vol=window_vol
                    )
                    # overlap check
                    skip = False
                    if hide_overlap:
                        for j_idx in active_obs_indices:
                            existing = obs[j_idx]
                            if is_price_overlap(ob_candidate.top, ob_candidate.btm, existing):
                                skip = True
                                break
                    # If not skipped, append OB and mark active
                    if not skip:
                        obs.append(ob_candidate)
                        active_obs_indices.append(len(obs) - 1)
                        # enforce show_last_x_ob: remove oldest from active if exceed
                        if len(active_obs_indices) > show_last_x_ob:
                            oldest_idx = active_obs_indices.pop(0)
                            # don't remove from obs list (we want to keep history), but mark as no longer active (violated_idx stays None)
                    # Signal decision: sell if bearish dominates by ratio. Emit signal even if OB was skipped due to overlap.
                    if ob_candidate.bearish_str / total >= min_strength_ratio:
                        sig_idx = t + 1 if (t + 1) < n else t
                        ob_index = (len(obs) - 1) if not skip else None
                        # Log why the sell signal was emitted
                        print(f"[SIG] SELL candidate t={t} sig_idx={sig_idx} price={float(df['close'].iat[sig_idx]):.6f} ob_idx={ob_index} skip={skip} bull={ob_candidate.bullish_str:.3f} bear={ob_candidate.bearish_str:.3f} total={total:.3f} create_idx={ob_candidate.create_idx} start_idx={ob_candidate.start_idx} top={ob_candidate.top:.6f} btm={ob_candidate.btm:.6f}")
                        # Diagnostic: show EMA and candidate entry for this signal
                        try:
                            cand_entry = float(df['low'].iat[sig_idx]) if entry_price_mode == "worst" else float(df['close'].iat[sig_idx])
                        except Exception:
                            cand_entry = None
                        try:
                            ema_val_dbg = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                        except Exception:
                            ema_val_dbg = None
                        if cand_entry is not None and ema_val_dbg is not None:
                            pct_diff = (cand_entry - ema_val_dbg) / ema_val_dbg
                        else:
                            pct_diff = None
                        print(f"[EMA DBG] SELL sig_idx={sig_idx} cand_entry={cand_entry} ema={ema_val_dbg} pct_diff={pct_diff} thr={float(globals().get('entry_diff_short_pct', entry_diff_short_pct))}")
                        # allow reversing signals: a bearish OB (normally a SELL) can be flipped to BUY
                        sig_kind = 'sell' if not reverse_signals else 'buy'
                        signals.append(Signal(
                            idx=sig_idx,
                            time=df['timestamp'].iat[sig_idx],
                            kind=sig_kind,
                            price=float(df['close'].iat[sig_idx]),
                            ob_idx=ob_index
                        ))
                        # Reset holding counters for existing positions of the SIGNAL side (refresh rationale)
                        try:
                            if holding_period_bars and holding_period_bars > 0:
                                refresh_side = 'long' if sig_kind == 'buy' else 'short'
                                for _pos in open_positions:
                                    if _pos.position_type == refresh_side:
                                        _pos.holding_counter = 0
                        except Exception:
                            pass

                        # Create position according to the emitted signal kind (supports reversed-signals)
                        if sig_kind == 'sell':
                            # Create short position if we have room for more positions
                            # Decide whether to open a position; optionally filter by EMA
                            # Determine candidate entry price first (used by EMA filter and later as the actual entry price)
                            if entry_price_mode == "worst":
                                cand_entry_price = float(df['low'].iat[sig_idx])
                            else:
                                cand_entry_price = float(df['close'].iat[sig_idx])

                            skip_open_due_to_ema = False
                            try:
                                ema_val = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                            except Exception:
                                ema_val = None

                            # For shorts we expect entry price to be at or above EMA; skip if candidate entry is too far below EMA
                            if ema_val is not None:
                                try:
                                    if cand_entry_price < ema_val * (1.0 - float(globals().get('entry_diff_short_pct', entry_diff_short_pct))):
                                        skip_open_due_to_ema = True
                                except Exception:
                                    skip_open_due_to_ema = False

                            if skip_open_due_to_ema:
                                print(f"[EMA SKIP] SHORT at sig_idx={sig_idx} cand_entry={cand_entry_price:.6f} ema={ema_val:.6f} threshold={float(globals().get('entry_diff_short_pct', entry_diff_short_pct)):.6f}")

                            if len(open_positions) < max_concurrent_positions and sig_idx < n and (not skip_open_due_to_ema):
                                ob_size = ob_candidate.top - ob_candidate.btm

                                # Sideways / low-volatility detection (single check)
                                skip_open_due_to_sideways = False
                                sideways_leverage_apply = 1.0
                                try:
                                    w = int(max(1, globals().get('atr_sideways_window', 20)))
                                    start_i = max(0, sig_idx - w + 1)
                                    atr_mean = float(atr.iloc[start_i:sig_idx+1].mean()) if sig_idx >= start_i else float(atr.iat[sig_idx])
                                    cand_price = cand_entry_price
                                    atr_pct = (atr_mean / cand_price) if (cand_price and cand_price > 0) else None
                                except Exception:
                                    atr_mean = None
                                    atr_pct = None

                                if atr_pct is not None:
                                    thr = float(globals().get('atr_sideways_threshold_pct', 0.001))
                                    act = str(globals().get('sideways_action', 'skip'))
                                    if atr_pct < thr:
                                        if act == 'skip':
                                            skip_open_due_to_sideways = True
                                        elif act == 'leverage':
                                            sideways_leverage_apply = float(globals().get('sideways_leverage_mult', 2.0))
                                        print(f"[SIDEWAYS DBG] SHORT sig_idx={sig_idx} atr_mean={atr_mean} atr_pct={atr_pct:.6f} thr={thr} action={act}")

                                if skip_open_due_to_sideways:
                                    print(f"[SIDEWAYS SKIP] SHORT at sig_idx={sig_idx} cand_entry={cand_entry_price:.6f} atr_pct={atr_pct:.6f} thr={thr}")
                                else:
                                    # Determine entry price based on mode
                                    entry_price = cand_entry_price

                                    # base distance = distance from entry to OB TOP (for shorts we expect SL above entry near/above OB top)
                                    raw_dist = max(0.0, ob_candidate.top - entry_price)
                                    # ATR-based floor to avoid vanishingly small SLs
                                    atr_val_here = float(atr.iat[sig_idx]) if (not np.isnan(atr.iat[sig_idx])) else 0.0
                                    atr_floor_val = atr_val_here * ob_min_sl_atr_mult
                                    pct_floor = entry_price * ob_min_sl_pct
                                    base_dist = max(raw_dist, atr_floor_val, pct_floor)

                                    # scale by multiplier to allow placing SL further/closer than the OB bound
                                    sl_distance = base_dist * stop_loss_multiplier
                                    stop_loss = entry_price + sl_distance   # SL is above entry for shorts
                                    # TP measured from entry using SL distance * TP multiplier
                                    tp_distance = sl_distance * take_profit_multiplier
                                    take_profit = entry_price - tp_distance

                                    # Calculate position size based on risk per trade
                                    capital_for_sizing = (initial_capital if use_fixed_capital else current_capital) * sideways_leverage_apply
                                    if sideways_leverage_apply != 1.0:
                                        print(f"[SIDEWAYS LEVERAGE] SHORT sig_idx={sig_idx} leverage_mult={sideways_leverage_apply}")
                                    capital_at_risk = capital_for_sizing * (risk_per_trade_percent / 100.0)
                                    # apply minimum per-unit risk floor to avoid extreme sizes
                                    min_risk_pct = 0.001  # 0.1% of entry price
                                    min_risk_per_unit = entry_price * min_risk_pct
                                    atr_half_floor = atr_val_here * 0.5  # half-ATR floor used as minimum per-unit risk
                                    # use the SL distance we actually set as the per-unit risk
                                    risk_per_unit = max(sl_distance, min_risk_per_unit, atr_half_floor)
                                    position_size = capital_at_risk / (risk_per_unit if risk_per_unit > 0 else 1.0)

                                    # Apply maximum position size constraint
                                    max_units = max_position_size_usd / entry_price if entry_price > 0 else position_size
                                    position_size = min(position_size, max_units)

                                    # Recalculate actual capital at risk based on constrained position size
                                    actual_capital_at_risk = position_size * risk_per_unit

                                    position = Position(
                                        entry_idx=sig_idx,
                                        entry_time=df['timestamp'].iat[sig_idx],
                                        entry_price=entry_price,
                                        position_type="short",
                                        stop_loss=stop_loss,
                                        take_profit=take_profit,
                                        ob_size=ob_size,
                                        sl_distance=sl_distance,
                                        tp_distance=tp_distance,
                                        position_size=position_size,
                                        capital_at_risk=actual_capital_at_risk,
                                        initial_stop_loss=stop_loss
                                    )
                                    positions.append(position)
                                    open_positions.append(position)
                                    # Log position sizing/entry details for debugging
                                    print(f"[OPEN] SHORT entry_idx={position.entry_idx} entry_price={position.entry_price:.6f} stop_loss={position.stop_loss:.6f} take_profit={position.take_profit:.6f} ob_size={position.ob_size:.6f} sl_distance={position.sl_distance:.6f} tp_distance={position.tp_distance:.6f} pos_size={position.position_size:.6f} cap_at_risk=${position.capital_at_risk:.2f} ob_idx={ob_index}")
                        else:
                            # Reversed: treat bearish OB as BUY signal -> open LONG using buy-branch logic
                            # compute candidate entry according to entry_price_mode appropriate for LONG
                            if entry_price_mode == "worst":
                                cand_entry_price_rev = float(df['high'].iat[sig_idx])
                            else:
                                cand_entry_price_rev = float(df['close'].iat[sig_idx])

                            skip_open_due_to_ema = False
                            try:
                                ema_val = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                            except Exception:
                                ema_val = None

                            # For longs, apply EMA filter similar to buy branch
                            if ema_val is not None:
                                try:
                                    if cand_entry_price_rev > ema_val * (1.0 + float(globals().get('entry_diff_long_pct', entry_diff_long_pct))):
                                        skip_open_due_to_ema = True
                                except Exception:
                                    skip_open_due_to_ema = False

                            if skip_open_due_to_ema:
                                print(f"[EMA SKIP] LONG(sig_rev) at sig_idx={sig_idx} cand_entry={cand_entry_price_rev:.6f} ema={ema_val:.6f} threshold={float(globals().get('entry_diff_long_pct', entry_diff_long_pct)):.6f}")

                            if len(open_positions) < max_concurrent_positions and sig_idx < n and (not skip_open_due_to_ema):
                                ob_size = ob_candidate.top - ob_candidate.btm
                                # Sideways detection for LONG
                                skip_open_due_to_sideways = False
                                sideways_leverage_apply = 1.0
                                try:
                                    w = int(max(1, globals().get('atr_sideways_window', 20)))
                                    start_i = max(0, sig_idx - w + 1)
                                    atr_mean = float(atr.iloc[start_i:sig_idx+1].mean()) if sig_idx >= start_i else float(atr.iat[sig_idx])
                                    cand_price = cand_entry_price_rev
                                    atr_pct = (atr_mean / cand_price) if (cand_price and cand_price > 0) else None
                                except Exception:
                                    atr_mean = None
                                    atr_pct = None

                                if atr_pct is not None:
                                    thr = float(globals().get('atr_sideways_threshold_pct', 0.001))
                                    act = str(globals().get('sideways_action', 'skip'))
                                    if atr_pct < thr:
                                        if act == 'skip':
                                            skip_open_due_to_sideways = True
                                        elif act == 'leverage':
                                            sideways_leverage_apply = float(globals().get('sideways_leverage_mult', 2.0))
                                        print(f"[SIDEWAYS DBG] LONG(sig_rev) sig_idx={sig_idx} atr_mean={atr_mean} atr_pct={atr_pct:.6f} thr={thr} action={act}")

                                if skip_open_due_to_sideways:
                                    print(f"[SIDEWAYS SKIP] LONG(sig_rev) at sig_idx={sig_idx} cand_entry={cand_entry_price_rev:.6f} atr_pct={atr_pct:.6f} thr={thr}")
                                else:
                                    entry_price = cand_entry_price_rev
                                    raw_dist = max(0.0, entry_price - ob_candidate.btm)
                                    atr_val_here = float(atr.iat[sig_idx]) if (not np.isnan(atr.iat[sig_idx])) else 0.0
                                    atr_floor_val = atr_val_here * ob_min_sl_atr_mult
                                    pct_floor = entry_price * ob_min_sl_pct
                                    base_dist = max(raw_dist, atr_floor_val, pct_floor)
                                    sl_distance = base_dist * stop_loss_multiplier
                                    stop_loss = entry_price - sl_distance
                                    tp_distance = sl_distance * take_profit_multiplier
                                    take_profit = entry_price + tp_distance

                                    capital_for_sizing = (initial_capital if use_fixed_capital else current_capital) * sideways_leverage_apply
                                    if sideways_leverage_apply != 1.0:
                                        print(f"[SIDEWAYS LEVERAGE] LONG(sig_rev) sig_idx={sig_idx} leverage_mult={sideways_leverage_apply}")
                                    capital_at_risk = capital_for_sizing * (risk_per_trade_percent / 100.0)
                                    min_risk_pct = 0.01
                                    min_risk_per_unit = entry_price * min_risk_pct
                                    atr_half_floor = atr_val_here * 0.5
                                    risk_per_unit = max(sl_distance, min_risk_per_unit, atr_half_floor)
                                    position_size = capital_at_risk / (risk_per_unit if risk_per_unit > 0 else 1.0)
                                    max_units = max_position_size_usd / entry_price if entry_price > 0 else position_size
                                    position_size = min(position_size, max_units)
                                    actual_capital_at_risk = position_size * risk_per_unit
                                    position = Position(
                                        entry_idx=sig_idx,
                                        entry_time=df['timestamp'].iat[sig_idx],
                                        entry_price=entry_price,
                                        position_type="long",
                                        stop_loss=stop_loss,
                                        take_profit=take_profit,
                                        ob_size=ob_size,
                                        sl_distance=sl_distance,
                                        tp_distance=tp_distance,
                                        position_size=position_size,
                                        capital_at_risk=actual_capital_at_risk,
                                        initial_stop_loss=stop_loss
                                    )
                                    positions.append(position)
                                    open_positions.append(position)
                                    print(f"[OPEN] LONG(sig_rev) entry_idx={position.entry_idx} entry_price={position.entry_price:.6f} stop_loss={position.stop_loss:.6f} take_profit={position.take_profit:.6f} ob_size={position.ob_size:.6f} sl_distance={position.sl_distance:.6f} tp_distance={position.tp_distance:.6f} pos_size={position.position_size:.6f} cap_at_risk=${position.capital_at_risk:.2f} ob_idx={ob_index}")

        # Check for break above last_top (bullish OB creation)
        if last_top is not None and (not last_top['crossed']):
            pivot_price = float(last_top['price'])
            atr_val = float(atr.iat[t]) if not np.isnan(atr.iat[t]) else 0.0
            required_break_high = pivot_price + (break_atr_mult * atr_val)
            if float(df['close'].iat[t]) > required_break_high:
                last_top['crossed'] = True
                # create bullish OB: find lowest red candle (close < open) in previous swing_length bars
                best_btm = np.inf
                best_top = np.nan
                best_idx = None
                best_vol = np.nan
                for i in range(1, swing_length + 1):
                    idx = t - i
                    if idx < 0:
                        break
                    if df['close'].iat[idx] < df['open'].iat[idx]:
                        if df['low'].iat[idx] < best_btm:
                            best_btm = df['low'].iat[idx]
                            best_top = df['high'].iat[idx]
                            best_idx = idx
                            best_vol = df['volume'].iat[idx]
                if best_idx is None:
                    # fallback: choose the minimum low regardless of color
                    for i in range(1, swing_length + 1):
                        idx = t - i
                        if idx < 0:
                            break
                        if df['low'].iat[idx] < best_btm:
                            best_btm = df['low'].iat[idx]
                            best_top = df['high'].iat[idx]
                            best_idx = idx
                            best_vol = df['volume'].iat[idx]

                if best_idx is not None:
                    # determine search window for OB geometry
                    # ensure search_w is an integer (user may pass a float like swing_length*0.25)
                    search_w = int(ob_search_window) if (ob_search_window is not None) else int(swing_length)
                    search_w = max(1, search_w)
                    half = search_w // 2
                    # ensure positional indices for iloc are integers
                    ws = max(0, int(best_idx) - int(half))
                    we = min(n - 1, int(best_idx) + int(half))
                    ob_top = float(highs.iloc[ws:we+1].max())
                    ob_btm = float(lows.iloc[ws:we+1].min())
                    window_vol = float(df['volume'].iloc[ws:we+1].sum())

                    bullish, bearish = calculate_strengths(df, best_idx, swing_length)
                    total = bullish + bearish
                    if total < min_total_volume:
                        total = min_total_volume
                    ob_candidate = OrderBlock(
                        kind='bullish',
                        top=ob_top,
                        btm=ob_btm,
                        start_idx=best_idx,
                        create_idx=t,
                        create_time=df['timestamp'].iat[t],
                        bullish_str=float(bullish),
                        bearish_str=float(bearish),
                        vol=window_vol
                    )
                    # overlap check
                    skip = False
                    if hide_overlap:
                        for j_idx in active_obs_indices:
                            existing = obs[j_idx]
                            if is_price_overlap(ob_candidate.top, ob_candidate.btm, existing):
                                skip = True
                                break
                    # If not skipped, append OB and mark active
                    if not skip:
                        obs.append(ob_candidate)
                        active_obs_indices.append(len(obs) - 1)
                        if len(active_obs_indices) > show_last_x_ob:
                            active_obs_indices.pop(0)
                    # Signal decision: buy if bullish dominates. Emit signal even if OB was skipped due to overlap.
                    if ob_candidate.bullish_str / total >= min_strength_ratio:
                        sig_idx = t + 1 if (t + 1) < n else t
                        ob_index = (len(obs) - 1) if not skip else None
                        # Log why the buy signal was emitted
                        print(f"[SIG] BUY candidate t={t} sig_idx={sig_idx} price={float(df['close'].iat[sig_idx]):.6f} ob_idx={ob_index} skip={skip} bull={ob_candidate.bullish_str:.3f} bear={ob_candidate.bearish_str:.3f} total={total:.3f} create_idx={ob_candidate.create_idx} start_idx={ob_candidate.start_idx} top={ob_candidate.top:.6f} btm={ob_candidate.btm:.6f}")
                        # Diagnostic: show EMA and candidate entry for this signal
                        try:
                            cand_entry = float(df['high'].iat[sig_idx]) if entry_price_mode == "worst" else float(df['close'].iat[sig_idx])
                        except Exception:
                            cand_entry = None
                        try:
                            ema_val_dbg = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                        except Exception:
                            ema_val_dbg = None
                        if cand_entry is not None and ema_val_dbg is not None:
                            pct_diff = (cand_entry - ema_val_dbg) / ema_val_dbg
                        else:
                            pct_diff = None
                        print(f"[EMA DBG] BUY sig_idx={sig_idx} cand_entry={cand_entry} ema={ema_val_dbg} pct_diff={pct_diff} thr={float(globals().get('entry_diff_long_pct', entry_diff_long_pct))}")
                        # allow reversing signals: a bullish OB (normally a BUY) can be flipped to SELL
                        sig_kind = 'buy' if not reverse_signals else 'sell'
                        signals.append(Signal(
                            idx=sig_idx,
                            time=df['timestamp'].iat[sig_idx],
                            kind=sig_kind,
                            price=float(df['close'].iat[sig_idx]),
                            ob_idx=ob_index
                        ))

                        # Reset holding counters for existing positions of the SIGNAL side (refresh rationale)
                        try:
                            if holding_period_bars and holding_period_bars > 0:
                                refresh_side = 'long' if sig_kind == 'buy' else 'short'
                                for _pos in open_positions:
                                    if _pos.position_type == refresh_side:
                                        _pos.holding_counter = 0
                        except Exception:
                            pass

                        # Create position according to the emitted signal kind (supports reversed-signals)
                        if sig_kind == 'buy':
                            # Create long position if we have room for more positions
                            # Decide whether to open a position; optionally filter by EMA
                            # Determine candidate entry price first (used by EMA filter and later as the actual entry price)
                            if entry_price_mode == "worst":
                                cand_entry_price = float(df['high'].iat[sig_idx])
                            else:
                                cand_entry_price = float(df['close'].iat[sig_idx])

                            skip_open_due_to_ema = False
                            try:
                                ema_val = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                            except Exception:
                                ema_val = None

                            # For longs we expect entry price to be at or below EMA; skip if candidate entry is too far above EMA
                            if ema_val is not None:
                                try:
                                    if cand_entry_price > ema_val * (1.0 + float(globals().get('entry_diff_long_pct', entry_diff_long_pct))):
                                        skip_open_due_to_ema = True
                                except Exception:
                                    skip_open_due_to_ema = False

                            if skip_open_due_to_ema:
                                print(f"[EMA SKIP] LONG at sig_idx={sig_idx} cand_entry={cand_entry_price:.6f} ema={ema_val:.6f} threshold={float(globals().get('entry_diff_long_pct', entry_diff_long_pct)):.6f}")

                            if len(open_positions) < max_concurrent_positions and sig_idx < n and (not skip_open_due_to_ema):
                                ob_size = ob_candidate.top - ob_candidate.btm
                                # Determine entry price based on mode
                                # Use the candidate entry price determined above
                                entry_price = cand_entry_price

                                # base distance = distance from entry to OB BOTTOM (for longs we expect SL below entry near/at OB bottom)
                                raw_dist = max(0.0, entry_price - ob_candidate.btm)
                                atr_val_here = float(atr.iat[sig_idx]) if (not np.isnan(atr.iat[sig_idx])) else 0.0
                                atr_floor_val = atr_val_here * ob_min_sl_atr_mult
                                pct_floor = entry_price * ob_min_sl_pct
                                base_dist = max(raw_dist, atr_floor_val, pct_floor)
                                sl_distance = base_dist * stop_loss_multiplier
                                stop_loss = entry_price - sl_distance   # SL below entry for longs
                                tp_distance = sl_distance * take_profit_multiplier
                                take_profit = entry_price + tp_distance

                                # Calculate position size based on risk per trade
                                # Sideways / low-volatility detection (single check) - mirror short-branch behavior
                                skip_open_due_to_sideways = False
                                sideways_leverage_apply = 1.0
                                try:
                                    w = int(max(1, globals().get('atr_sideways_window', 20)))
                                    start_i = max(0, sig_idx - w + 1)
                                    atr_mean = float(atr.iloc[start_i:sig_idx+1].mean()) if sig_idx >= start_i else float(atr.iat[sig_idx])
                                    cand_price = entry_price
                                    atr_pct = (atr_mean / cand_price) if (cand_price and cand_price > 0) else None
                                except Exception:
                                    atr_mean = None
                                    atr_pct = None

                                if atr_pct is not None:
                                    thr = float(globals().get('atr_sideways_threshold_pct', 0.001))
                                    act = str(globals().get('sideways_action', 'skip'))
                                    if atr_pct < thr:
                                        if act == 'skip':
                                            skip_open_due_to_sideways = True
                                        elif act == 'leverage':
                                            sideways_leverage_apply = float(globals().get('sideways_leverage_mult', 2.0))
                                        print(f"[SIDEWAYS DBG] LONG sig_idx={sig_idx} atr_mean={atr_mean} atr_pct={atr_pct:.6f} thr={thr} action={act}")

                                if skip_open_due_to_sideways:
                                    print(f"[SIDEWAYS SKIP] LONG at sig_idx={sig_idx} cand_entry={entry_price:.6f} atr_pct={atr_pct:.6f} thr={thr}")
                                    continue

                                capital_for_sizing = (initial_capital if use_fixed_capital else current_capital) * sideways_leverage_apply
                                if sideways_leverage_apply != 1.0:
                                    print(f"[SIDEWAYS LEVERAGE] LONG sig_idx={sig_idx} leverage_mult={sideways_leverage_apply}")
                                capital_at_risk = capital_for_sizing * (risk_per_trade_percent / 100.0)
                                # apply minimum per-unit risk floor to avoid extreme sizes
                                min_risk_pct = 0.01  # 1% of entry price
                                min_risk_per_unit = entry_price * min_risk_pct
                                atr_half_floor = atr_val_here * 0.5  # half-ATR floor
                                risk_per_unit = max(sl_distance, min_risk_per_unit, atr_half_floor)
                                position_size = capital_at_risk / (risk_per_unit if risk_per_unit > 0 else 1.0)

                                # Apply maximum position size constraint
                                max_units = max_position_size_usd / entry_price if entry_price > 0 else position_size
                                position_size = min(position_size, max_units)

                                # Recalculate actual capital at risk based on constrained position size
                                actual_capital_at_risk = position_size * risk_per_unit

                                position = Position(
                                    entry_idx=sig_idx,
                                    entry_time=df['timestamp'].iat[sig_idx],
                                    entry_price=entry_price,
                                    position_type="long",
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    ob_size=ob_size,
                                    sl_distance=sl_distance,
                                    tp_distance=tp_distance,
                                    position_size=position_size,
                                    capital_at_risk=actual_capital_at_risk,
                                    initial_stop_loss=stop_loss
                                )
                                positions.append(position)
                                open_positions.append(position)
                                # Log position sizing/entry details for debugging
                                print(f"[OPEN] LONG entry_idx={position.entry_idx} entry_price={position.entry_price:.6f} stop_loss={position.stop_loss:.6f} take_profit={position.take_profit:.6f} ob_size={position.ob_size:.6f} sl_distance={position.sl_distance:.6f} tp_distance={position.tp_distance:.6f} pos_size={position.position_size:.6f} cap_at_risk=${position.capital_at_risk:.2f} ob_idx={ob_index}")
                        else:
                            # Reversed: treat bullish OB as SELL signal -> open SHORT using short-branch logic
                            if entry_price_mode == "worst":
                                cand_entry_price_rev = float(df['low'].iat[sig_idx])
                            else:
                                cand_entry_price_rev = float(df['close'].iat[sig_idx])

                            skip_open_due_to_ema = False
                            try:
                                ema_val = float(ema.iat[sig_idx]) if (not pd.isna(ema.iat[sig_idx])) else None
                            except Exception:
                                ema_val = None

                            # For shorts, apply EMA filter similar to sell branch
                            if ema_val is not None:
                                try:
                                    if cand_entry_price_rev < ema_val * (1.0 - float(globals().get('entry_diff_short_pct', entry_diff_short_pct))):
                                        skip_open_due_to_ema = True
                                except Exception:
                                    skip_open_due_to_ema = False

                            if skip_open_due_to_ema:
                                print(f"[EMA SKIP] SHORT(sig_rev) at sig_idx={sig_idx} cand_entry={cand_entry_price_rev:.6f} ema={ema_val:.6f} threshold={float(globals().get('entry_diff_short_pct', entry_diff_short_pct)):.6f}")

                            if len(open_positions) < max_concurrent_positions and sig_idx < n and (not skip_open_due_to_ema):
                                ob_size = ob_candidate.top - ob_candidate.btm

                                # Sideways / low-volatility detection (single check)
                                skip_open_due_to_sideways = False
                                sideways_leverage_apply = 1.0
                                try:
                                    w = int(max(1, globals().get('atr_sideways_window', 20)))
                                    start_i = max(0, sig_idx - w + 1)
                                    atr_mean = float(atr.iloc[start_i:sig_idx+1].mean()) if sig_idx >= start_i else float(atr.iat[sig_idx])
                                    cand_price = cand_entry_price_rev
                                    atr_pct = (atr_mean / cand_price) if (cand_price and cand_price > 0) else None
                                except Exception:
                                    atr_mean = None
                                    atr_pct = None

                                if atr_pct is not None:
                                    thr = float(globals().get('atr_sideways_threshold_pct', 0.001))
                                    act = str(globals().get('sideways_action', 'skip'))
                                    if atr_pct < thr:
                                        if act == 'skip':
                                            skip_open_due_to_sideways = True
                                        elif act == 'leverage':
                                            sideways_leverage_apply = float(globals().get('sideways_leverage_mult', 2.0))
                                        print(f"[SIDEWAYS DBG] SHORT(sig_rev) sig_idx={sig_idx} atr_mean={atr_mean} atr_pct={atr_pct:.6f} thr={thr} action={act}")

                                if skip_open_due_to_sideways:
                                    print(f"[SIDEWAYS SKIP] SHORT(sig_rev) at sig_idx={sig_idx} cand_entry={cand_entry_price_rev:.6f} atr_pct={atr_pct:.6f} thr={thr}")
                                else:
                                    # Determine entry price based on mode
                                    entry_price = cand_entry_price_rev

                                    # base distance = distance from entry to OB TOP (for shorts we expect SL above entry near/above OB top)
                                    raw_dist = max(0.0, ob_candidate.top - entry_price)
                                    # ATR-based floor to avoid vanishingly small SLs
                                    atr_val_here = float(atr.iat[sig_idx]) if (not np.isnan(atr.iat[sig_idx])) else 0.0
                                    atr_floor_val = atr_val_here * ob_min_sl_atr_mult
                                    pct_floor = entry_price * ob_min_sl_pct
                                    base_dist = max(raw_dist, atr_floor_val, pct_floor)

                                    # scale by multiplier to allow placing SL further/closer than the OB bound
                                    sl_distance = base_dist * stop_loss_multiplier
                                    stop_loss = entry_price + sl_distance   # SL is above entry for shorts
                                    # TP measured from entry using SL distance * TP multiplier
                                    tp_distance = sl_distance * take_profit_multiplier
                                    take_profit = entry_price - tp_distance

                                    # Calculate position size based on risk per trade
                                    capital_for_sizing = (initial_capital if use_fixed_capital else current_capital) * sideways_leverage_apply
                                    if sideways_leverage_apply != 1.0:
                                        print(f"[SIDEWAYS LEVERAGE] SHORT(sig_rev) sig_idx={sig_idx} leverage_mult={sideways_leverage_apply}")
                                    capital_at_risk = capital_for_sizing * (risk_per_trade_percent / 100.0)
                                    # apply minimum per-unit risk floor to avoid extreme sizes
                                    min_risk_pct = 0.001  # 0.1% of entry price
                                    min_risk_per_unit = entry_price * min_risk_pct
                                    atr_half_floor = atr_val_here * 0.5  # half-ATR floor used as minimum per-unit risk
                                    # use the SL distance we actually set as the per-unit risk
                                    risk_per_unit = max(sl_distance, min_risk_per_unit, atr_half_floor)
                                    position_size = capital_at_risk / (risk_per_unit if risk_per_unit > 0 else 1.0)

                                    # Apply maximum position size constraint
                                    max_units = max_position_size_usd / entry_price if entry_price > 0 else position_size
                                    position_size = min(position_size, max_units)

                                    # Recalculate actual capital at risk based on constrained position size
                                    actual_capital_at_risk = position_size * risk_per_unit

                                    position = Position(
                                        entry_idx=sig_idx,
                                        entry_time=df['timestamp'].iat[sig_idx],
                                        entry_price=entry_price,
                                        position_type="short",
                                        stop_loss=stop_loss,
                                        take_profit=take_profit,
                                        ob_size=ob_size,
                                        sl_distance=sl_distance,
                                        tp_distance=tp_distance,
                                        position_size=position_size,
                                        capital_at_risk=actual_capital_at_risk,
                                        initial_stop_loss=stop_loss
                                    )
                                    positions.append(position)
                                    open_positions.append(position)
                                    print(f"[OPEN] SHORT(sig_rev) entry_idx={position.entry_idx} entry_price={position.entry_price:.6f} stop_loss={position.stop_loss:.6f} take_profit={position.take_profit:.6f} ob_size={position.ob_size:.6f} sl_distance={position.sl_distance:.6f} tp_distance={position.tp_distance:.6f} pos_size={position.position_size:.6f} cap_at_risk=${position.capital_at_risk:.2f} ob_idx={ob_index}")

        # After potential creation, optionally update active OB strengths with conservative
        # forward reinforcement: if the completed candle's body lies fully inside the OB
        # (i.e. body_low >= ob.btm and body_high <= ob.top) then count its volume toward
        # bullish_str or bearish_str depending on candle direction. This avoids counting
        # touching/violating wicks and prevents reinforcement on bars that actually violate
        # the OB.
        last_body_low = float(min(df['open'].iat[t], df['close'].iat[t]))
        last_body_high = float(max(df['open'].iat[t], df['close'].iat[t]))
        last_vol = float(df['volume'].iat[t]) if 'volume' in df.columns else 0.0

        for ob_i in active_obs_indices:
            ob = obs[ob_i]
            if ob.violated_idx is not None:
                continue
            try:
                # conservative test: candle body fully inside OB bounds
                body_inside = (last_body_low >= ob.btm) and (last_body_high <= ob.top)
            except Exception:
                body_inside = False

            if body_inside and last_vol > 0:
                # classify candle direction by body
                if df['close'].iat[t] >= df['open'].iat[t]:
                    ob.bullish_str = float(ob.bullish_str) + last_vol
                else:
                    ob.bearish_str = float(ob.bearish_str) + last_vol
                ob.vol = float(ob.bullish_str) + float(ob.bearish_str)

        # Now run the existing violation detection and maintain active_obs_indices
        still_active = []
        for ob_i in active_obs_indices:
            ob = obs[ob_i]
            if ob.violated_idx is not None:
                # already violated
                continue
            # violation conditions
            if ob.kind == 'bullish':
                violated = (violation_type == 'Wick' and float(df['low'].iat[t]) < ob.btm) or \
                           (violation_type == 'Close' and float(df['close'].iat[t]) < ob.btm)
            else:  # bearish
                violated = (violation_type == 'Wick' and float(df['high'].iat[t]) > ob.top) or \
                           (violation_type == 'Close' and float(df['close'].iat[t]) > ob.top)
            if violated:
                ob.violated_idx = t
                ob.violated_time = df['timestamp'].iat[t]
                # remove from active list
            else:
                still_active.append(ob_i)
        active_obs_indices = still_active
    
    # Close any remaining open positions at the last bar
    for pos in open_positions:
        last_idx = n - 1
        pnl_dollars = finalize_position_exit(
            pos,
            exit_idx=last_idx,
            exit_time_ms=df['timestamp'].iat[last_idx],
            exit_price=float(df['close'].iat[last_idx]),
            exit_reason="end_of_data",
            commission_percent=commission_percent,
            slippage_percent=slippage_percent
        )
        current_capital += pnl_dollars

    # append final equity point at end
    if n > 0:
        equity_ts.append({'time': int(df['timestamp'].iat[-1]), 'equity': float(current_capital)})

    equity_df = pd.DataFrame(equity_ts)
    try:
        print(f"[TRAILING STATS] activations={n_trailing_activations} updates={n_trailing_updates} updates_skipped={n_trailing_updates_skipped} exits={n_trailing_exits}")
    except Exception:
        pass
    return obs, signals, positions, equity_df

# =========================
# Visualization
# =========================
def plot_with_obs(df: pd.DataFrame, obs: List[OrderBlock], signals: List[Signal], positions: List[Position] = None, initial_capital: float = None, figsize=(14,8)):
    """
    Plot close price line, draw OB rectangles from ob.start time to create/violation time (or end),
    plot buy/sell markers, and show position management with SL/TP levels.
    """
    fig, ax = plt.subplots(figsize=figsize)
    times = pd.to_datetime(df['timestamp'], unit='ms')
    ax.plot(times, df['close'], color='black', linewidth=1.2, label='close', zorder=3)

    # prepare map from ob -> left/right in datetime
    end_time = times.iloc[-1]
    for i, ob in enumerate(obs):
        left_time = pd.to_datetime(df['timestamp'].iat[ob.start_idx], unit='ms')
        right_time = pd.to_datetime(ob.violated_time, unit='ms') if ob.violated_time is not None else end_time
        width = mdates.date2num(right_time) - mdates.date2num(left_time)
        # rectangle bottom and height
        bottom = ob.btm
        height = ob.top - ob.btm
        color = 'teal' if ob.kind == 'bullish' else 'orangered'
        alpha = 0.18
        rect = Rectangle(
            (mdates.date2num(left_time), bottom),
            width, height,
            facecolor=color, edgecolor='none', alpha=alpha, zorder=1
        )
        ax.add_patch(rect)
        # draw border
        ax.add_patch(Rectangle(
            (mdates.date2num(left_time), bottom),
            width, height,
            facecolor='none', edgecolor=color, linewidth=0.8, alpha=0.6, zorder=2
        ))
        # annotate strength
        mid_x = left_time + (right_time - left_time) / 2
        #label = f"B:{ob.bullish_str:.0f}\nS:{ob.bearish_str:.0f}"
        label = '' #nill for now to make it cleaner
        ax.text(mid_x, ob.top, label, ha='center', va='bottom', fontsize=8, color=color, zorder=5)

    # plot signals
    buy_times = [s.time for s in signals if s.kind == 'buy']
    buy_prices = [s.price for s in signals if s.kind == 'buy']
    sell_times = [s.time for s in signals if s.kind == 'sell']
    sell_prices = [s.price for s in signals if s.kind == 'sell']

    if buy_times:
        ax.scatter(pd.to_datetime(buy_times, unit='ms'), buy_prices, marker='^', color='green', s=80, zorder=10, label='buy')
    if sell_times:
        ax.scatter(pd.to_datetime(sell_times, unit='ms'), sell_prices, marker='v', color='red', s=80, zorder=10, label='sell')

    # Plot positions with SL and TP levels
    if positions:
        for i, pos in enumerate(positions):
            entry_time = pd.to_datetime(pos.entry_time, unit='ms')
            exit_time = pd.to_datetime(pos.exit_time, unit='ms') if pos.exit_time is not None else times.iloc[-1]
            
            # Determine color based on position type and outcome
            if pos.pnl is not None and pos.pnl > 0:
                pos_color = 'green'
                alpha_val = 0.3
            elif pos.pnl is not None and pos.pnl < 0:
                pos_color = 'red'
                alpha_val = 0.3
            else:
                pos_color = 'gray'
                alpha_val = 0.2
            
            # Draw horizontal lines for SL and TP
            sl_color = 'orange' if pos.trailing_active else 'red'
            sl_label = 'Trailing SL' if (i == 0 and pos.trailing_active) else ('SL' if i == 0 else '')
            ax.hlines(pos.stop_loss, entry_time, exit_time, colors=sl_color, linestyles='dashed', 
                     linewidth=1.5, alpha=0.7, zorder=4, label=sl_label)
            
            # Show initial SL if trailing was activated
            if pos.initial_stop_loss is not None and pos.trailing_active and pos.initial_stop_loss != pos.stop_loss:
                ax.hlines(pos.initial_stop_loss, entry_time, exit_time, colors='red', linestyles='dotted', 
                         linewidth=1.0, alpha=0.4, zorder=4, label='Initial SL' if i == 0 else '')
            
            ax.hlines(pos.take_profit, entry_time, exit_time, colors='green', linestyles='dashed', 
                     linewidth=1.5, alpha=0.7, zorder=4, label='TP' if i == 0 else '')
            ax.hlines(pos.entry_price, entry_time, exit_time, colors=pos_color, linestyles='solid', 
                     linewidth=1.0, alpha=0.5, zorder=4)
            
            # Mark exit point
            if pos.exit_time is not None and pos.exit_price is not None:
                exit_marker = 'X'
                if pos.exit_reason == 'TP':
                    exit_color = 'green'
                elif pos.exit_reason == 'trailing_stop':
                    exit_color = 'orange'
                elif pos.exit_reason == 'SL':
                    exit_color = 'red'
                else:
                    exit_color = 'gray'
                    
                ax.scatter([exit_time], [pos.exit_price], marker=exit_marker, 
                          color=exit_color, s=100, zorder=11, edgecolors='black', linewidths=1)
                
                # Add PnL annotation
                if pos.pnl is not None:
                    pnl_text = f"{pos.exit_reason}\n"
                    if pos.pnl_dollars is not None:
                        pnl_text += f"${pos.pnl_dollars:.2f} ({pos.pnl_percent:.1f}%)"
                    else:
                        pnl_text += f"{pos.pnl_percent:.1f}%"
                    ax.annotate(pnl_text, xy=(exit_time, pos.exit_price), 
                               xytext=(10, -10 if pos.position_type == 'long' else 10),
                               textcoords='offset points', fontsize=8, 
                               color=exit_color, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=exit_color))

    # formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    ax.set_xlabel('time')
    ax.set_ylabel('price')
    
    # Add performance stats if positions exist
    if positions:
        closed_positions = [p for p in positions if p.pnl is not None]
        if closed_positions:
            total_trades = len(closed_positions)
            winning_trades = len([p for p in closed_positions if p.pnl > 0])
            losing_trades = len([p for p in closed_positions if p.pnl < 0])
            trailing_stop_exits = len([p for p in closed_positions if p.exit_reason == 'trailing_stop'])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            # sum of per-trade percent values (not the same as portfolio ROI)
            total_pnl_percent_sum = sum(p.pnl_percent for p in closed_positions if p.pnl_percent is not None)
            total_pnl_dollars = sum(p.pnl_dollars for p in closed_positions if p.pnl_dollars is not None)
            avg_win = sum(p.pnl_percent for p in closed_positions if p.pnl is not None and p.pnl > 0) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum(p.pnl_percent for p in closed_positions if p.pnl is not None and p.pnl < 0) / losing_trades if losing_trades > 0 else 0

            # Portfolio ROI computed from dollar PnL relative to initial capital (preferred metric)
            roi_percent = (total_pnl_dollars / initial_capital * 100) if (initial_capital is not None and initial_capital > 0) else None

            stats_text = f"Trades: {total_trades} | Win: {winning_trades} | Loss: {losing_trades} | Win Rate: {win_rate:.1f}%"
            if trailing_stop_exits > 0:
                stats_text += f" | Trailing: {trailing_stop_exits}"
            # show both the (misleading) sum of per-trade percents and the portfolio ROI when available
            if roi_percent is not None:
                stats_text += f"\nPortfolio ROI: {roi_percent:.2f}% (${total_pnl_dollars:.2f}) | Sum(per-trade %): {total_pnl_percent_sum:.2f}% | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%"
            else:
                stats_text += f"\nTotal PnL (sum per-trade %): {total_pnl_percent_sum:.2f}% (${total_pnl_dollars:.2f}) | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Price with Volumetric Order Blocks, Signals, and Position Management')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_portfolio_equity(df: pd.DataFrame, equity_df: pd.DataFrame, bh_equity: Optional[pd.Series] = None, figsize=(14,6)):
    """
    Plot portfolio equity over time for the strategy (equity_df) and optional buy-and-hold
    series (bh_equity). equity_df expected to have columns ['time','equity'] where time is ms.
    bh_equity should be a pd.Series aligned to `df` (same index) containing mark-to-market equity values.
    """
    if equity_df is None or len(equity_df) == 0:
        print("No strategy equity data to plot.")
        return

    times_strat = pd.to_datetime(equity_df['time'], unit='ms')
    strat_eq = equity_df['equity'].astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times_strat, strat_eq, label='Strategy Equity', color='tab:blue', linewidth=1.5)

    if bh_equity is not None and len(bh_equity) == len(df):
        times_bh = pd.to_datetime(df['timestamp'], unit='ms')
        ax.plot(times_bh, bh_equity.astype(float), label='Buy & Hold Equity', color='tab:orange', linestyle='--')

    ax.set_xlabel('time')
    ax.set_ylabel('portfolio value')
    ax.set_title('Portfolio Equity Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    # Compute and display stats (total return, max drawdown, Sharpe) for strategy and BH
    try:
        # estimate periods_per_year from df timestamps (median delta)
        if len(df) >= 2:
            deltas = pd.to_datetime(df['timestamp'], unit='ms').diff().dropna().dt.total_seconds()
            median_delta = deltas.median() if len(deltas) > 0 else 60.0
            seconds_per_year = 365.25 * 24 * 3600
            periods_per_year = int(max(1, round(seconds_per_year / median_delta)))
        else:
            periods_per_year = 252

        # Strategy stats
        strat_eq = equity_df['equity'].astype(float).reset_index(drop=True)
        strat_ret = (float(strat_eq.iat[-1]) / float(strat_eq.iat[0]) - 1.0) * 100.0 if len(strat_eq) > 0 else None
        strat_dd = compute_drawdown(strat_eq) if len(strat_eq) > 0 else {}
        strat_sharpe = annualized_sharpe(strat_eq, periods_per_year=periods_per_year)

        # Buy-and-hold stats
        bh_ret = None
        bh_dd = {}
        bh_sharpe = None
        if bh_equity is not None and len(bh_equity) > 0:
            bh_s = bh_equity.astype(float).reset_index(drop=True)
            bh_ret = (float(bh_s.iat[-1]) / float(bh_s.iat[0]) - 1.0) * 100.0
            bh_dd = compute_drawdown(bh_s)
            bh_sharpe = annualized_sharpe(bh_s, periods_per_year=periods_per_year)

        # build stats text
        lines = []
        lines.append('Strategy')
        if strat_ret is not None:
            lines.append(f"Total: {strat_ret:.2f}%")
        if strat_dd:
            lines.append(f"Max DD: {strat_dd['max_drawdown_pct']:.2f}% ({strat_dd['duration_bars']} bars)")
        else:
            lines.append('Max DD: N/A')
        lines.append(f"Sharpe: {str(strat_sharpe) if strat_sharpe is not None else 'N/A'}")
        lines.append('')
        lines.append('Buy & Hold')
        if bh_ret is not None:
            lines.append(f"Total: {bh_ret:.2f}%")
        if bh_dd:
            lines.append(f"Max DD: {bh_dd.get('max_drawdown_pct', 'N/A'):.2f}% ({bh_dd.get('duration_bars', 0)} bars)")
        else:
            lines.append('Max DD: N/A')
        lines.append(f"Sharpe: {str(bh_sharpe) if bh_sharpe is not None else 'N/A'}")

        stats_text = '\n'.join(lines)
        # place stats box in upper right
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
    except Exception as e:
        # don't fail plotting if stats computation errors
        print(f"Warning: could not compute/display portfolio stats: {e}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def compute_simple_buy_and_short(df: pd.DataFrame, initial_capital: float,
                                 commission_percent: float, slippage_percent: float):
    """
    Compute a simple buy-and-hold long and a simple short over the same period.

    - Entry: first bar close
    - Exit: last bar close
    - Use full capital to size the position (position_size = initial_capital / entry_price)
    - Fees model matches `finalize_position_exit`: fees per unit = price * ((commission+slippage)/100)

    Returns dicts for 'long' and 'short' with keys: entry_price, exit_price, position_size,
    pnl_dollars, pnl_percent, roi_percent, total_fees_dollars
    """
    out_long = None
    out_short = None
    try:
        if df is None or len(df) < 1:
            return None, None
        entry_price = float(df['close'].iat[0])
        exit_price = float(df['close'].iat[len(df) - 1])

        # protect against zero price
        if entry_price == 0:
            return None, None

        position_size = initial_capital / entry_price

        # fees per unit at entry + exit (percent variables are treated as percent, matching file semantics)
        fees_per_unit = entry_price * ((commission_percent + slippage_percent) / 100.0)
        fees_per_unit += exit_price * ((commission_percent + slippage_percent) / 100.0)
        total_fees = fees_per_unit * position_size

        # Long
        pnl_long = (exit_price - entry_price) * position_size - total_fees
        pnl_long_percent = (pnl_long / initial_capital) * 100 if initial_capital != 0 else None
        roi_long = pnl_long_percent

        out_long = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl_dollars': pnl_long,
            'pnl_percent_of_capital': pnl_long_percent,
            'roi_percent': roi_long,
            'total_fees_dollars': total_fees
        }

        # Short (sell at entry, buy back at exit)
        pnl_short = (entry_price - exit_price) * position_size - total_fees
        pnl_short_percent = (pnl_short / initial_capital) * 100 if initial_capital != 0 else None
        roi_short = pnl_short_percent

        out_short = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl_dollars': pnl_short,
            'pnl_percent_of_capital': pnl_short_percent,
            'roi_percent': roi_short,
            'total_fees_dollars': total_fees
        }

        return out_long, out_short
    except Exception:
        return None, None


def compute_drawdown(equity_series: pd.Series) -> dict:
    """
    Compute drawdown metrics from an equity_series (indexed by time or integer index).
    Returns dict with: peak_val, trough_val, max_drawdown (fraction), max_drawdown_pct,
    max_drawdown_start_idx, max_drawdown_end_idx, duration_bars
    """
    if equity_series is None or len(equity_series) == 0:
        return {}
    # ensure float Series
    eq = equity_series.astype(float).reset_index(drop=True)
    running_max = eq.cummax()
    drawdowns = (eq - running_max) / running_max
    # most negative drawdown
    min_dd = drawdowns.min()
    if pd.isna(min_dd):
        return {}
    end_idx = int(drawdowns.idxmin())
    # find start (last index before end_idx where eq == running_max)
    if end_idx == 0:
        start_idx = 0
    else:
        # running_max at end_idx is the peak value
        peak_val = running_max.iat[end_idx]
        # find first index where running_max == peak_val going backwards
        peak_idxs = running_max[:end_idx+1][running_max[:end_idx+1] == peak_val]
        start_idx = int(peak_idxs.index[-1]) if len(peak_idxs) > 0 else 0

    duration = end_idx - start_idx
    peak_val = float(running_max.iat[start_idx])
    trough_val = float(eq.iat[end_idx])
    return {
        'peak_val': peak_val,
        'trough_val': trough_val,
        'max_drawdown': float(min_dd),
        'max_drawdown_pct': float(min_dd * 100.0),
        'start_idx': start_idx,
        'end_idx': end_idx,
        'duration_bars': int(duration)
    }


def annualized_sharpe(equity_series: pd.Series, periods_per_year: int = 252 * 24 * 60) -> Optional[float]:
    """
    Compute annualized Sharpe ratio from equity series. Assumes equity_series is mark-to-market
    portfolio equity (dollars). Converts to returns (periodic simple returns) and computes
    annualized mean / std. periods_per_year defaults to 1-min bars (~252 trading days * 24h * 60m)
    but user can supply e.g., 252 for daily.
    Returns None if not computable.
    """
    if equity_series is None or len(equity_series) < 2:
        return None
    eq = equity_series.astype(float).reset_index(drop=True)
    # simple returns
    returns = eq.pct_change().dropna()
    if returns.std(ddof=0) == 0 or returns.std(ddof=0) is None:
        return None
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=0)
    # annualize
    ann_mean = mean_ret * periods_per_year
    ann_std = std_ret * np.sqrt(periods_per_year)
    if ann_std == 0:
        return None
    sharpe = ann_mean / ann_std
    return float(sharpe)


def analyze_position_outcomes(positions: List[Position]):
    """
    Analyze closed positions and print diagnostics comparing SL vs TP exits,
    average distances, and fees impact.
    """
    closed = [p for p in positions if p.exit_reason is not None]
    if not closed:
        print("No closed positions to analyze SL vs TP outcomes.")
        return

    sl_exits = [p for p in closed if p.exit_reason == 'SL']
    tp_exits = [p for p in closed if p.exit_reason == 'TP']
    trailing_exits = [p for p in closed if p.exit_reason == 'trailing_stop']

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    avg_sl_loss_dollars = safe_mean([p.pnl_dollars for p in sl_exits if p.pnl_dollars is not None])
    avg_sl_loss_pct = safe_mean([p.pnl_percent for p in sl_exits if p.pnl_percent is not None])

    avg_tp_gain_dollars = safe_mean([p.pnl_dollars for p in tp_exits if p.pnl_dollars is not None])
    avg_tp_gain_pct = safe_mean([p.pnl_percent for p in tp_exits if p.pnl_percent is not None])

    avg_fees = safe_mean([p.total_fees_dollars for p in closed if p.total_fees_dollars is not None])

    avg_sl_distance = safe_mean([p.sl_distance for p in closed if p.sl_distance is not None])
    avg_tp_distance = safe_mean([p.tp_distance for p in closed if p.tp_distance is not None])

    print("\n--- SL vs TP Diagnostics ---")
    print(f"Total closed trades: {len(closed)} | SL exits: {len(sl_exits)} | TP exits: {len(tp_exits)} | Trailing exits: {len(trailing_exits)}")
    print(f"Average SL distance (price units): {avg_sl_distance:.6f} | Average TP distance (price units): {avg_tp_distance:.6f}")
    print(f"Average fees per trade: ${avg_fees:.2f}")
    if sl_exits:
        print(f"Avg SL PnL: ${avg_sl_loss_dollars:.2f} ({avg_sl_loss_pct:.2f}%) over {len(sl_exits)} trades")
    else:
        print("No SL exits to report.")
    if tp_exits:
        print(f"Avg TP PnL: ${avg_tp_gain_dollars:.2f} ({avg_tp_gain_pct:.2f}%) over {len(tp_exits)} trades")
    else:
        print("No TP exits to report.")

    # Quick observation to help user tune strategy
    if sl_exits and tp_exits and abs(avg_sl_loss_dollars) > avg_tp_gain_dollars:
        print("Observation: average dollar loss on SL exits is larger than average dollar gain on TP exits. Consider: \n"
              " - increasing take_profit_multiplier,\n"
              " - reducing stop_loss_multiplier, or\n"
              " - reviewing fee/slippage assumptions (they may be eating small gains).")
    print("----------------------------\n")


def inspect_signal_context(df: pd.DataFrame, obs: List[OrderBlock], signals: List[Signal], *,
                           signal_list_index: Optional[int] = None,
                           ob_index: Optional[int] = None,
                           neighbor_ob_radius: int = 1,
                           candle_pad: int = 5) -> dict:
    """
    Print and return context around a given signal or OB index to help debugging.

    Provide either `signal_list_index` (index into the `signals` list) or `ob_index` (index into `obs`).
    The function will print the selected signal, the referenced OB, `neighbor_ob_radius` OBs to
    the left and right (if present), and `candle_pad` candles around the OB.create_idx (breakout bar).

    Returns a dict with keys: 'signal', 'selected_ob', 'neighbor_obs', 'candles'.
    """
    # resolve ob_index from signal if needed
    selected_signal = None
    if signal_list_index is not None:
        try:
            selected_signal = signals[signal_list_index]
            ob_index = selected_signal.ob_idx
        except Exception as e:
            print(f"inspect_signal_context: invalid signal_list_index: {signal_list_index} -> {e}")
            return {}

    if ob_index is None:
        print("inspect_signal_context: no ob_index provided and signal has no ob_idx (skipped OB).")
        return {'signal': selected_signal}

    out = {'signal': selected_signal, 'selected_ob': None, 'neighbor_obs': [], 'candles': []}

    # bounds check
    if ob_index < 0 or ob_index >= len(obs):
        print(f"inspect_signal_context: ob_index {ob_index} out of range (0..{len(obs)-1})")
        return out

    sel_ob = obs[ob_index]
    out['selected_ob'] = sel_ob

    print(f"\n=== INSPECT OB #{ob_index} ===")
    print(f"kind={sel_ob.kind} start_idx={sel_ob.start_idx} create_idx={sel_ob.create_idx} top={sel_ob.top} btm={sel_ob.btm}")
    print(f"bull={sel_ob.bullish_str} bear={sel_ob.bearish_str} vol={sel_ob.vol}")

    # neighbor OBs
    left = max(0, ob_index - neighbor_ob_radius)
    right = min(len(obs) - 1, ob_index + neighbor_ob_radius)
    for i in range(left, right + 1):
        if i == ob_index:
            continue
        o = obs[i]
        out['neighbor_obs'].append((i, o))
        print(f"NEIGHBOR OB #{i}: kind={o.kind} start_idx={o.start_idx} create_idx={o.create_idx} top={o.top} btm={o.btm} bull={o.bullish_str} bear={o.bearish_str}")

    # print candles around create_idx
    ci = int(sel_ob.create_idx)
    n = len(df)
    start_c = max(0, ci - candle_pad)
    end_c = min(n - 1, ci + candle_pad)
    print(f"\nCandles around create_idx={ci} (indices {start_c}..{end_c}):\nindex | time | open high low close volume")
    for idx in range(start_c, end_c + 1):
        r = df.iloc[idx]
        ts = pd.to_datetime(r['timestamp'], unit='ms') if 'timestamp' in df.columns else idx
        print(f"{idx:5d} | {ts} | {r['open']:.6f} {r['high']:.6f} {r['low']:.6f} {r['close']:.6f} vol={r.get('volume', np.nan)}")
        out['candles'].append({'idx': idx, 'time': ts, 'open': r['open'], 'high': r['high'], 'low': r['low'], 'close': r['close'], 'volume': r.get('volume', np.nan)})

    # If a signal is available, print the signal candle and relation to OB
    if selected_signal is not None:
        sig_idx = int(selected_signal.idx)
        sig_price = float(selected_signal.price)
        print(f"\nSignal (list idx={signal_list_index}) -> sig_idx={sig_idx} sig_price={sig_price:.6f} ob_idx={selected_signal.ob_idx}")
        print(f"Signal price < OB bottom? {sig_price < sel_ob.btm} | Signal price > OB top? {sig_price > sel_ob.top}")

    print("=== END INSPECT ===\n")
    return out





#%% for testing if the above cells are not changed
# =========================
# PARAMETERS (change here)
# =========================

coin = "ETHUSDT"                  # trading pair
timeframe = '15m'                  # timeframe for candles
start = "2025-11-30"                # start date for data collection
end = "2025-12-02"                  # end date for data collection


reverse_signals = False              # if True, invert buy/sell signals (for testing)
show_last_x_ob = 2                 # how many recent OBs to keep (used for plotting limit)
violation_type = "Close"            # "Wick" or "Close"
hide_overlap = False                # if True, skip creating overlapping OBs
min_total_volume = 1e-7            # guard to avoid division by zero

# Detection tunables
swing_length = 30                   # number of candles used for pivot swingLength
break_atr_mult = 0.02         # require breakout > 0.2 * ATR
ob_search_window = swing_length*0.3        # build OB from bars surrounding the selected candle
min_strength_ratio = 0.3    # min ratio (dominant side) to emit a signal

# OB SL floor tunables
# atr_period_mult is a multiplier applied to `swing_length` to compute the default ATR window
atr_period_mult = 1.2        # ATR window multiplier of swing_length (atr_period = round(swing_length * atr_period_mult))
ob_min_sl_atr_mult = 2    # ensure SL at least this * ATR
ob_min_sl_pct = 0.01       # ensure SL at least this fraction of entry price

# ATR selection for SL floors / range measurements
# Options: 'highlow_sma' (legacy per-candle high-low mean),
#          'maxmin_range' (rolling max(high)-min(low) structural range),
#          'close_range' (rolling max(close)-min(close) settled-price range)
atr_method = 'highlow_sma'
# fraction of the structural range to use when using maxmin_range or close_range (1.0 = full range)
range_percent = 1

# Sideways / low-volatility handling
# If recent ATR (mean over window) / entry_price is below `atr_sideways_threshold_pct` then
# take the action in `sideways_action` which can be 'skip' or 'leverage'.
# If 'skip' the entry is skipped. If 'leverage' the sizing capital_for_sizing is multiplied by
# `sideways_leverage_mult` (e.g., 2 means twice the capital -> larger position) for that trade.
atr_sideways_window = swing_length*0.3           # number of bars to average ATR over for sideways detection
atr_sideways_threshold_pct = 0.002  # threshold as fraction of price (0.001 = 0.1%)
sideways_action = 'skip'            # 'skip' or 'leverage'
sideways_leverage_mult = 0.5        # multiplier applied to capital_for_sizing when action == 'leverage'

# Position management parameters
stop_loss_multiplier = 1.2         # SL distance = OB_size × this multiplier (SL placed INSIDE the OB)
take_profit_multiplier = 15       # TP distance = SL_distance × this multiplier
max_concurrent_positions = 3       # maximum number of open positions at once
trailing_stop_activation = 2     # activate trailing stop after profit reaches this multiple of SL distance
trailing_stop_percent = 1        # trail stop by this % below highest (longs) or above lowest (shorts)
trailing_stop_buffer_candles = 10   # wait this many candles after activation before allowing trailing stop exit
trailing_stop_update_threshold = 0.4  # only update trailing stop if price moves by this % since last update

# Holding period: if >0, close position after this many bars without a reinforcing same-side signal
# Example: for 15m bars, 48 -> 12 hours
holding_period_bars = 480

# Capital management parameters
initial_capital = 5000.0          # starting capital in USD (or your currency)
risk_per_trade_percent = 3.0       # risk this % of capital per trade (used for position sizing)
use_fixed_capital = True           # if True, always use initial capital for position sizing (no compounding)
max_position_size_usd = 100000.0   # maximum position value in USD (prevents unrealistic large positions)

# Entry and fees
entry_price_mode = "Close"         # "close" = use close price, "worst" = use high for longs/low for shorts (more conservative)
commission_percent = 0           # trading fee percentage (e.g., 0.1% = 0.001 per trade)
slippage_percent = 0            # slippage percentage (e.g., 0.05% = 0.0005 per trade)


# EMA-based entry filter tuning
ema_period = swing_length*0.4                     # lookback for EMA used to filter entries
# If entry price is this fraction above EMA (e.g., 0.02 = 2%), skip LONG entries
entry_diff_long_pct = 0.02
# If entry price is this fraction below EMA (e.g., 0.02 = 2%), skip SHORT entries
entry_diff_short_pct = 0.03



# Placeholder for data, it's commented out intentionally to prevent repeated fetching
data = None  #data is already in cache so this is intentionally

# =========================
# Example / entry point
# =========================
if __name__ == "__main__":

    # Import BinanceDataCollector here so the module can be imported elsewhere
    # without requiring `requests` to be installed at import-time.
    if data is None:
        try:
            from binance_collector import BinanceDataCollector
            
        except Exception as e:
            print("BinanceDataCollector not available:", e)
            BinanceDataCollector = None

        if BinanceDataCollector is None:
            #data = None
            print('using old data for now')
        else:
            collector = BinanceDataCollector(futures=True)
            data = collector.fetch_candles(
                coin, timeframe,
                start_date= start,
                end_date= end
            )

    if data is None:
        print("No data collected from Binance. Replace with your own DataFrame named `df` to run detection.")
    else:
        df = data.sort_values('timestamp').reset_index(drop=True)

        obs, signals, positions, equity = detect_order_blocks(
            df,
            swing_length=swing_length,
            hide_overlap=hide_overlap,
            show_last_x_ob=show_last_x_ob,
            violation_type=violation_type,
            min_strength_ratio=min_strength_ratio,
            stop_loss_multiplier=stop_loss_multiplier,
            take_profit_multiplier=take_profit_multiplier,
            max_concurrent_positions=max_concurrent_positions,
            initial_capital=initial_capital,
            risk_per_trade_percent=risk_per_trade_percent,
            trailing_stop_activation=trailing_stop_activation,
            trailing_stop_percent=trailing_stop_percent,
            trailing_stop_buffer_candles=trailing_stop_buffer_candles,
            trailing_stop_update_threshold=trailing_stop_update_threshold,
            use_fixed_capital=use_fixed_capital,
            max_position_size_usd=max_position_size_usd,
            entry_price_mode=entry_price_mode,
            commission_percent=commission_percent,
            slippage_percent=slippage_percent
            ,break_atr_mult=break_atr_mult
            ,ob_search_window=ob_search_window
            ,ob_min_sl_atr_mult=ob_min_sl_atr_mult
            ,ob_min_sl_pct=ob_min_sl_pct
            ,holding_period_bars=holding_period_bars
            ,reverse_signals=reverse_signals
        )

        print(f"Detected {len(obs)} OBs, {len(signals)} signals, {len(positions)} positions")
        print(f"\nInitial Capital: ${initial_capital:.2f}")
        print(f"Risk per Trade: {risk_per_trade_percent}%")
        print(f"Position Sizing Mode: {'Fixed Capital (No Compounding)' if use_fixed_capital else 'Compounding Capital'}")
        print(f"Max Position Size: ${max_position_size_usd:.2f}")
        print(f"Entry Price Mode: {entry_price_mode.upper()}")
        print(f"Commission: {commission_percent}% | Slippage: {slippage_percent}%")
        print(f"Trailing Stop Activation: {trailing_stop_activation}x SL distance")
        print(f"Trailing Stop Percent: {trailing_stop_percent}% from peak")
        print(f"Trailing Stop Buffer: {trailing_stop_buffer_candles} candles")
        print(f"Trailing Stop Update Threshold: {trailing_stop_update_threshold}%")
        # expose the new OB tunables
        print(f"Break ATR Multiplier (break_atr_mult): {break_atr_mult}")
        print(f"OB Search Window (ob_search_window): {ob_search_window}")
        print(f"OB min SL ATR multiplier (ob_min_sl_atr_mult): {ob_min_sl_atr_mult}")
        print(f"OB min SL pct floor (ob_min_sl_pct): {ob_min_sl_pct}")
        
        print("\n=== SIGNALS ===")
        for i, s in enumerate(signals):
            print(f"{i}: {s.kind.upper()} at {s.price:.2f} (time: {pd.to_datetime(s.time, unit='ms')})")
        
        print("\n=== POSITIONS ===")
        for i, pos in enumerate(positions):
            entry_dt = pd.to_datetime(pos.entry_time, unit='ms')
            exit_dt = pd.to_datetime(pos.exit_time, unit='ms') if pos.exit_time else 'Open'
            print(f"\n{i+1}. {pos.position_type.upper()} Position:")
            print(f"   Entry: {pos.entry_price:.2f} at {entry_dt}")
            print(f"   Position Size: {pos.position_size:.6f} units (risk: ${pos.capital_at_risk:.2f})")
            print(f"   Stop Loss: {pos.stop_loss:.2f} (distance: {pos.sl_distance:.2f})")
            print(f"   Take Profit: {pos.take_profit:.2f} (distance: {pos.tp_distance:.2f})")
            print(f"   OB Size: {pos.ob_size:.2f}")
            print(f"   Risk/Reward: 1:{pos.tp_distance/pos.sl_distance:.2f}")
            if pos.trailing_active:
                print(f"   Trailing Stop: ACTIVE (initial SL: {pos.initial_stop_loss:.2f})")
            if pos.exit_time:
                print(f"   Exit: {pos.exit_price:.2f} at {exit_dt} (Reason: {pos.exit_reason})")
                if pos.pnl_dollars is not None:
                    fee_per_unit_str = f"${pos.total_cost_per_unit:.6f}" if pos.total_cost_per_unit is not None else "N/A"
                    total_fees_str = f"${pos.total_fees_dollars:.2f}" if pos.total_fees_dollars is not None else "N/A"
                    print(f"   Fees: {fee_per_unit_str} per unit | Total fees: {total_fees_str}")
                    print(f"   PnL: ${pos.pnl_dollars:.2f} ({pos.pnl_percent:.2f}%)")
                else:
                    print(f"   PnL: {pos.pnl:.2f} ({pos.pnl_percent:.2f}%)")
        
        # Calculate summary statistics
        closed_positions = [p for p in positions if p.pnl is not None]
        if closed_positions:
            total_trades = len(closed_positions)
            winning_trades = [p for p in closed_positions if p.pnl > 0]
            losing_trades = [p for p in closed_positions if p.pnl < 0]
            trailing_stop_trades = [p for p in closed_positions if p.exit_reason == 'trailing_stop']
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            total_pnl_pct = sum(p.pnl_percent for p in closed_positions)
            total_pnl_dollars = sum(p.pnl_dollars for p in closed_positions if p.pnl_dollars is not None)
            total_fees_paid = sum(p.total_fees_dollars for p in closed_positions if p.total_fees_dollars is not None)
            final_capital = initial_capital + total_pnl_dollars
            roi = (total_pnl_dollars / initial_capital * 100) if initial_capital > 0 else 0
            
            print("\n=== PERFORMANCE SUMMARY ===")
            print(f"Initial Capital: ${initial_capital:.2f}")
            print(f"Total Fees Paid: ${total_fees_paid:.2f}")
            print(f"Final Capital: ${final_capital:.2f}")
            print(f"Total P&L: ${total_pnl_dollars:.2f} ({roi:.2f}% ROI)")
            print(f"\nTotal Trades: {total_trades}")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Trailing Stop Exits: {len(trailing_stop_trades)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Average PnL per Trade: {total_pnl_pct/total_trades:.2f}%")
            if winning_trades:
                avg_win = sum(p.pnl_percent for p in winning_trades) / len(winning_trades)
                avg_win_dollars = sum(p.pnl_dollars for p in winning_trades if p.pnl_dollars is not None) / len(winning_trades)
                print(f"Average Win: {avg_win:.2f}% (${avg_win_dollars:.2f})")
            if losing_trades:
                avg_loss = sum(p.pnl_percent for p in losing_trades) / len(losing_trades)
                avg_loss_dollars = sum(p.pnl_dollars for p in losing_trades if p.pnl_dollars is not None) / len(losing_trades)
                print(f"Average Loss: {avg_loss:.2f}% (${avg_loss_dollars:.2f})")

    # === SL/TP DIAGNOSTICS ===
    analyze_position_outcomes(positions)

    # === SIMPLE BUY & HOLD / SIMPLE SHORT COMPARISON ===
    long_bh, short_bh = compute_simple_buy_and_short(df, initial_capital, commission_percent, slippage_percent)
    print("\n=== BUY & HOLD vs SIMPLE SHORT (same period) ===")
    if long_bh is None or short_bh is None:
        print("Not enough data to compute buy & hold / short comparison.")
    else:
        print("\n-- Buy & Hold (Long) --")
        print(f"Entry price: {long_bh['entry_price']:.2f} | Exit price: {long_bh['exit_price']:.2f}")
        print(f"Position size (units): {long_bh['position_size']:.6f}")
        print(f"Total fees: ${long_bh['total_fees_dollars']:.2f}")
        print(f"P&L: ${long_bh['pnl_dollars']:.2f} ({long_bh['roi_percent']:.2f}% of capital)")

        print("\n-- Simple Short --")
        print(f"Entry price (short open): {short_bh['entry_price']:.2f} | Exit price (cover): {short_bh['exit_price']:.2f}")
        print(f"Position size (units): {short_bh['position_size']:.6f}")
        print(f"Total fees: ${short_bh['total_fees_dollars']:.2f}")
        print(f"P&L: ${short_bh['pnl_dollars']:.2f} ({short_bh['roi_percent']:.2f}% of capital)")

        # --- Portfolio metrics: strategy equity vs buy-and-hold equity ---
        try:
            # equity DataFrame from detect_order_blocks: columns ['time','equity']
            strat_eq_df = equity.copy()
            strat_eq_df = strat_eq_df.sort_values('time').reset_index(drop=True)
            strat_equity = strat_eq_df['equity'] if 'equity' in strat_eq_df.columns else pd.Series()

            # Build buy-and-hold equity series (mark-to-market). Subtract entry fees upfront.
            bh_entry = float(long_bh['entry_price'])
            bh_pos_size = float(long_bh['position_size'])
            # entry fee per unit
            entry_fee_per_unit = bh_entry * ((commission_percent + slippage_percent) / 100.0)
            total_entry_fees = entry_fee_per_unit * bh_pos_size
            # initial effective capital after paying entry fees
            bh_initial_net = initial_capital - total_entry_fees
            bh_equity = bh_initial_net + bh_pos_size * (df['close'].astype(float) - bh_entry)
            # align indices/times with strategy equity if possible
            # compute periods_per_year from median timestamp delta
            if len(df) >= 2:
                deltas = pd.to_datetime(df['timestamp'], unit='ms').diff().dropna().dt.total_seconds()
                median_delta = deltas.median() if len(deltas) > 0 else 60.0
                seconds_per_year = 365.25 * 24 * 3600
                periods_per_year = int(max(1, round(seconds_per_year / median_delta)))
            else:
                periods_per_year = 252

            # Strategy metrics
            strat_dd = compute_drawdown(strat_equity) if (strat_equity is not None and len(strat_equity) > 0) else {}
            strat_sharpe = annualized_sharpe(strat_equity, periods_per_year=periods_per_year)
            strat_total_return = None
            if len(strat_equity) > 0:
                strat_total_return = (float(strat_equity.iat[-1]) / float(strat_equity.iat[0]) - 1.0) * 100.0

            # Buy-and-hold metrics
            bh_dd = compute_drawdown(bh_equity) if (bh_equity is not None and len(bh_equity) > 0) else {}
            bh_sharpe = annualized_sharpe(bh_equity, periods_per_year=periods_per_year)
            bh_total_return = (float(bh_equity.iat[-1]) / float(bh_equity.iat[0]) - 1.0) * 100.0

            print("\n--- Portfolio Metrics ---")
            print("Strategy (mark-to-market equity):")
            if strat_total_return is not None:
                print(f"  Total return: {strat_total_return:.2f}%")
            if strat_dd:
                print(f"  Max Drawdown: {strat_dd['max_drawdown_pct']:.2f}% (duration {strat_dd['duration_bars']} bars)")
            else:
                print("  Max Drawdown: N/A")
            print(f"  Annualized Sharpe: {str(strat_sharpe) if strat_sharpe is not None else 'N/A'}")

            print("\nBuy & Hold (mark-to-market equity):")
            print(f"  Total return: {bh_total_return:.2f}%")
            if bh_dd:
                print(f"  Max Drawdown: {bh_dd['max_drawdown_pct']:.2f}% (duration {bh_dd['duration_bars']} bars)")
            else:
                print("  Max Drawdown: N/A")
            print(f"  Annualized Sharpe: {str(bh_sharpe) if bh_sharpe is not None else 'N/A'}")

        except Exception as e:
            print(f"Error computing portfolio metrics: {e}")

    # Plot (if you have more data, you'll see rectangles & markers)
    plot_with_obs(df, obs, signals, positions, initial_capital=initial_capital)

    # Plot portfolio equity after the signal/OB plot
    try:
        # strat_eq_df and bh_equity were computed above when metrics were printed; if not present, try to derive
        if 'strat_eq_df' in locals():
            s_eq_df = strat_eq_df
        else:
            s_eq_df = equity

        # bh_equity may exist from earlier computation; otherwise build a simple buy-and-hold series here
        if 'bh_equity' in locals():
            bh_eq = bh_equity
        else:
            # fallback: compute buy-and-hold mark-to-market (no upfront fee subtraction)
            try:
                entry_price = float(df['close'].iat[0])
                bh_pos_size = initial_capital / entry_price if entry_price > 0 else 0
                bh_eq = initial_capital + bh_pos_size * (df['close'].astype(float) - entry_price)
            except Exception:
                bh_eq = None

        plot_portfolio_equity(df, s_eq_df, bh_eq)
    except Exception as e:
        print(f"Could not plot portfolio equity: {e}")

# %%

