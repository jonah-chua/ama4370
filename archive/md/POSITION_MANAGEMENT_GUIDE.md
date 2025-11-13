# Position Management Guide

## Overview
This document explains the position management system added to the order block trading strategy.

## New Features

### 1. Position Tracking
The system now tracks complete position lifecycle:
- **Entry**: When a signal is generated
- **Stop Loss (SL)**: Automatically calculated based on order block size
- **Take Profit (TP)**: Calculated as a multiple of stop loss distance
- **Exit**: Tracks when and why positions close (SL hit, TP hit, or end of data)
- **PnL**: Calculates profit/loss in both absolute and percentage terms

### 2. Position Parameters

#### Stop Loss Calculation
- **For Long Positions (Bullish OB)**:
  ```
  OB Size = OB_high - OB_low
  SL Distance = OB_size × stop_loss_multiplier
  Stop Loss = OB_low - SL_distance
  ```

- **For Short Positions (Bearish OB)**:
  ```
  OB Size = OB_high - OB_low
  SL Distance = OB_size × stop_loss_multiplier
  Stop Loss = OB_high + SL_distance
  ```

#### Take Profit Calculation
- **For Long Positions**:
  ```
  TP Distance = SL_distance × take_profit_multiplier
  Take Profit = Entry_price + TP_distance
  ```

- **For Short Positions**:
  ```
  TP Distance = SL_distance × take_profit_multiplier
  Take Profit = Entry_price - TP_distance
  ```

### 3. Example (Your Bearish Scenario)

Given:
- Bearish OB with high = $100, low = $95
- Entry (short) at $95
- `stop_loss_multiplier = 1.1`
- `take_profit_multiplier = 3.0`

Calculations:
```
OB Size = $100 - $95 = $5
SL Distance = $5 × 1.1 = $5.5
Stop Loss = $100 + $5.5 = $105.5 (above OB high for short)
TP Distance = $5.5 × 3.0 = $16.5
Take Profit = $95 - $16.5 = $78.5 (below entry for short)
Risk/Reward Ratio = 1:3
```

### 4. Configuration Parameters

Add these to the PARAMETERS section:

```python
stop_loss_multiplier = 1.1         # SL distance = OB_size × this multiplier
take_profit_multiplier = 3.0       # TP distance = SL_distance × this multiplier
max_concurrent_positions = 1       # maximum number of open positions at once
```

### 5. New Data Structures

#### Position Class
```python
@dataclass
class Position:
    entry_idx: int                    # Bar index of entry
    entry_time: pd.Timestamp          # Entry timestamp
    entry_price: float                # Entry price
    position_type: str                # "long" or "short"
    stop_loss: float                  # Stop loss price level
    take_profit: float                # Take profit price level
    ob_size: float                    # Size of the order block
    sl_distance: float                # Distance from entry to SL
    tp_distance: float                # Distance from entry to TP
    exit_idx: Optional[int]           # Bar index of exit
    exit_time: Optional[pd.Timestamp] # Exit timestamp
    exit_price: Optional[float]       # Exit price
    exit_reason: Optional[str]        # "SL", "TP", or "end_of_data"
    pnl: Optional[float]              # Profit/loss in price units
    pnl_percent: Optional[float]      # Profit/loss as percentage
```

### 6. Visualization Enhancements

The chart now displays:
- **SL Levels**: Red dashed horizontal lines
- **TP Levels**: Green dashed horizontal lines
- **Entry Levels**: Colored solid horizontal lines (green for profitable, red for losses)
- **Exit Points**: 'X' markers (green for TP, red for SL, orange for other)
- **PnL Annotations**: Shows exit reason and percentage gain/loss
- **Performance Stats Box**: Summary of trades, win rate, total PnL, etc.

### 7. Console Output

The script now prints detailed information:
- List of all signals
- Detailed position information (entry, exit, SL, TP, PnL)
- Performance summary (total trades, win rate, average win/loss)

## Usage Tips

### Adjusting Risk Management

1. **Conservative Setup** (Lower Risk):
   ```python
   stop_loss_multiplier = 0.5      # Tighter stop
   take_profit_multiplier = 2.0    # 1:2 risk/reward
   ```

2. **Aggressive Setup** (Higher Risk):
   ```python
   stop_loss_multiplier = 2.0      # Wider stop
   take_profit_multiplier = 5.0    # 1:5 risk/reward
   ```

3. **Your Current Setup** (Balanced):
   ```python
   stop_loss_multiplier = 1.1      # Slightly wider than OB
   take_profit_multiplier = 3.0    # 1:3 risk/reward
   ```

### Multiple Positions

To allow multiple concurrent positions:
```python
max_concurrent_positions = 3  # Allow up to 3 positions at once
```

### Position Management Logic

The system checks for SL/TP hits on every bar:
- **Long Position**: Closes if `low <= SL` or `high >= TP`
- **Short Position**: Closes if `high >= SL` or `low <= TP`
- Positions are checked BEFORE new signals are evaluated
- Any open positions at the end of data are closed at the last bar's close price

## Suggestions for Further Improvement

1. **Trailing Stop Loss**: Move SL in profit to lock in gains
2. **Partial Exits**: Close 50% at TP1, let rest run to TP2
3. **Dynamic Position Sizing**: Risk a fixed % of capital per trade
4. **Time-based Exits**: Close position after X bars if no SL/TP hit
5. **Break-even Stop**: Move SL to entry after certain profit threshold
6. **Risk/Reward Filter**: Only take trades with minimum R:R ratio
7. **Maximum Drawdown Limit**: Stop trading after X consecutive losses
8. **Commission/Slippage**: Add realistic trading costs

## Risk/Reward Analysis

The current setup with `stop_loss_multiplier = 1.1` and `take_profit_multiplier = 3.0` gives you a 1:3 risk/reward ratio, which means:
- You need to win only 25% of trades to break even (before costs)
- At 40% win rate, you'd be profitable
- At 50% win rate, you'd have excellent returns

This is a favorable risk/reward profile for a trading strategy!
