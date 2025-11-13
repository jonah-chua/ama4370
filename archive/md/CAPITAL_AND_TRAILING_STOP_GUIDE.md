# Capital Management & Trailing Stop Loss Guide

## New Features Added

### 1. Capital Management & Position Sizing

Your strategy now includes **realistic capital management** with proper position sizing based on risk per trade.

#### How It Works:

**Position Size Calculation:**
```
Capital at Risk = Current Capital × (Risk Per Trade % / 100)
Risk Per Unit = |Entry Price - Stop Loss|
Position Size = Capital at Risk / Risk Per Unit
```

**Example:**
- Initial Capital: $10,000
- Risk Per Trade: 2%
- Entry Price: $100
- Stop Loss: $95
- Risk Per Unit: $5

```
Capital at Risk = $10,000 × 0.02 = $200
Position Size = $200 / $5 = 40 units (e.g., 40 shares or 0.4 BTC)
```

This means:
- If SL is hit, you lose exactly $200 (2% of capital)
- If TP is hit with 3:1 R:R, you gain $600 (6% of capital)
- Capital compounds over time as you win/lose trades

#### Configuration:
```python
initial_capital = 10000.0          # Starting capital in USD
risk_per_trade_percent = 2.0       # Risk 2% of capital per trade
```

### 2. Trailing Stop Loss

The strategy now includes a **trailing stop loss** that locks in profits as the price moves in your favor.

#### How It Works:

**Activation:**
- The trailing stop activates when your unrealized profit reaches a threshold
- Threshold = `trailing_stop_activation × SL_distance`
- Default: 1.0 (activates when profit = 1× the initial risk)

**Trailing Behavior:**
- **For Long Positions:**
  - Tracks the highest price reached
  - Stop loss trails by: `highest_price - (SL_distance × trailing_stop_callback)`
  - Only moves up, never down

- **For Short Positions:**
  - Tracks the lowest price reached
  - Stop loss trails by: `lowest_price + (SL_distance × trailing_stop_callback)`
  - Only moves down, never up

#### Example (Long Position):

```
Entry: $100
Initial SL: $95 (SL distance = $5)
Trailing Activation: 1.0 × $5 = $5 profit
Trailing Callback: 0.5 × $5 = $2.5

Price Movement:
1. Price rises to $105 → Profit = $5 → Trailing ACTIVATES
2. Trailing SL = $105 - $2.5 = $102.50 (locks in $2.50 profit)
3. Price rises to $110 → Highest = $110
4. Trailing SL = $110 - $2.5 = $107.50 (locks in $7.50 profit)
5. Price falls to $107.00 → Trailing SL ($107.50) is HIT
6. Exit at $107.50 with $7.50 profit (even though price continued to fall)
```

#### Configuration:
```python
trailing_stop_activation = 1.0     # Activate after 1× SL distance profit
trailing_stop_callback = 0.5       # Trail by 0.5× SL distance from peak
```

### 3. Enhanced Visualization

The chart now shows:

1. **Initial Stop Loss**: Red dotted line (if trailing was activated)
2. **Current Stop Loss**: 
   - Red dashed line (not trailing)
   - Orange dashed line (actively trailing)
3. **Take Profit**: Green dashed line
4. **Exit Points**:
   - Green X = Take Profit hit
   - Orange X = Trailing Stop hit
   - Red X = Initial Stop Loss hit
   - Gray X = Other reason
5. **PnL Labels**: Shows exit reason and $ amount + percentage

### 4. Enhanced Performance Reporting

**Console Output Now Includes:**
- Initial and Final Capital
- Total P&L in dollars
- ROI (Return on Investment %)
- Position size and capital at risk per trade
- Number of trailing stop exits
- Average win/loss in both % and $

**Chart Stats Box Shows:**
- Number of trailing stop exits
- Total PnL in both % and dollars

## Configuration Examples

### Conservative Setup (2% Risk, Tight Trailing)
```python
initial_capital = 10000.0
risk_per_trade_percent = 2.0       # Risk 2% per trade
stop_loss_multiplier = 1.1         # Tight stop
take_profit_multiplier = 3.0       # 1:3 R:R
trailing_stop_activation = 0.5     # Activate early (at 50% of risk distance)
trailing_stop_callback = 0.3       # Tight trailing (30% of risk distance)
```

### Aggressive Setup (5% Risk, Wide Trailing)
```python
initial_capital = 10000.0
risk_per_trade_percent = 5.0       # Risk 5% per trade (higher risk!)
stop_loss_multiplier = 2.0         # Wide stop
take_profit_multiplier = 5.0       # 1:5 R:R
trailing_stop_activation = 1.5     # Activate later (at 150% of risk distance)
trailing_stop_callback = 0.8       # Wide trailing (80% of risk distance)
```

### Your Current Setup (Balanced)
```python
initial_capital = 10000.0
risk_per_trade_percent = 2.0       # Risk 2% per trade
stop_loss_multiplier = 1.1         # Slightly wider than OB
take_profit_multiplier = 3.0       # 1:3 R:R
trailing_stop_activation = 1.0     # Activate at 1× risk distance
trailing_stop_callback = 0.5       # Trail by 50% of risk distance
```

## Understanding the Parameters

### `trailing_stop_activation`
- **Lower values (0.5-0.8)**: More protective, activates sooner
  - Pros: Locks in profits earlier
  - Cons: May get stopped out of good trades too early

- **Higher values (1.5-2.0)**: More aggressive, lets profits run
  - Pros: Captures bigger moves
  - Cons: May give back more profit before trailing activates

### `trailing_stop_callback`
- **Lower values (0.3-0.5)**: Tighter trailing stop
  - Pros: Protects profits better, smaller drawdowns from peak
  - Cons: May get stopped out on minor retracements

- **Higher values (0.7-1.0)**: Wider trailing stop
  - Pros: Gives trades more room to breathe
  - Cons: May give back more profit before exit

## Example Trade Walkthrough

**Setup:**
- Capital: $10,000
- Risk per trade: 2% = $200
- Bearish OB detected: High=$100, Low=$95
- SL multiplier: 1.1
- TP multiplier: 3.0
- Trailing activation: 1.0
- Trailing callback: 0.5

**Position Calculation:**
```
OB Size = $100 - $95 = $5
SL Distance = $5 × 1.1 = $5.50
Entry (short) = $95
Stop Loss = $100 + $5.50 = $105.50
TP Distance = $5.50 × 3.0 = $16.50
Take Profit = $95 - $16.50 = $78.50

Risk Per Unit = $105.50 - $95 = $10.50
Position Size = $200 / $10.50 = 19.05 units
```

**Scenario 1: Normal Take Profit**
- Price falls to $78.50 → TP hit
- Profit = ($95 - $78.50) × 19.05 = $314.33
- ROI = 3.14% on capital

**Scenario 2: Trailing Stop Success**
- Price falls to $89.50 → Profit = $5.50, trailing ACTIVATES
- Initial trailing SL = $89.50 + ($5.50 × 0.5) = $92.25
- Price continues to $85 → Lowest = $85
- Trailing SL = $85 + $2.75 = $87.75
- Price rebounds to $88 → Trailing SL ($87.75) is HIT
- Profit = ($95 - $87.75) × 19.05 = $138.11
- ROI = 1.38% on capital
- Locked in 50% of potential TP profit!

**Scenario 3: Initial Stop Loss**
- Price rises to $105.50 → Initial SL hit
- Loss = ($105.50 - $95) × 19.05 = $200
- ROI = -2.0% on capital
- Exactly as planned (risk management working)

## Tips for Optimization

1. **Backtest Different Settings**: Test various trailing parameters on historical data
2. **Market Conditions**: 
   - Trending markets → Wider trailing (0.6-0.8 callback)
   - Choppy markets → Tighter trailing (0.3-0.5 callback)
3. **Volatility Adjustment**: Consider using ATR-based multipliers instead of fixed values
4. **Position Sizing**: Never risk more than 2-3% per trade for sustainable growth
5. **Monitor Trailing Exits**: If too many profitable trades exit via trailing stop too early, increase the callback parameter

## Risk Warning

- Proper position sizing is crucial for long-term survival
- 2% risk per trade allows you to survive 50 consecutive losses
- 5% risk per trade = only 20 consecutive losses to blow the account
- Always test strategies on demo/paper trading first
- Past performance doesn't guarantee future results

## Next Steps to Consider

1. **Monte Carlo Simulation**: Run 1000s of randomized trade sequences
2. **Maximum Drawdown Tracking**: Monitor largest peak-to-trough decline
3. **Sharpe Ratio Calculation**: Risk-adjusted return metric
4. **Walk-Forward Analysis**: Test on rolling time windows
5. **Multiple Timeframe Filters**: Confirm trades with higher timeframes
