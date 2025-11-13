# 📊 ProfitView Order Blocks - Tracking & Logging Features

## Overview

The ProfitView `Trading` class now includes comprehensive tracking and logging capabilities. All data is automatically logged during live trading and can be accessed via webhooks.

---

## 🗄️ Data Tracked

### 1. **Trade History** (`self.trade_history`)
Every completed trade is logged with:
- `side`: 'long' or 'short'
- `entry`: Entry price
- `exit`: Exit price
- `entry_time`: Unix timestamp (ms)
- `exit_time`: Unix timestamp (ms)
- `pnl_pct`: Profit/Loss percentage
- `pnl_usd`: Profit/Loss in USD
- `reason`: 'SL', 'TP', or other exit reason
- `bars_held`: Number of bars position was open
- `capital_after`: Capital after this trade

**Example:**
```python
{
    'side': 'long',
    'entry': 50000.0,
    'exit': 51000.0,
    'entry_time': 1699401600000,
    'exit_time': 1699405200000,
    'pnl_pct': 2.0,
    'pnl_usd': 200.0,
    'reason': 'TP',
    'bars_held': 12,
    'capital_after': 10200.0
}
```

---

### 2. **Equity Curve** (`self.equity_curve`)
Tracks capital over time (last 1000 points):
- `time`: Unix timestamp
- `price`: Current market price
- `capital`: Current capital
- `bar`: Bar index

**Example:**
```python
{
    'time': 1699401600000,
    'price': 50000.0,
    'capital': 10000.0,
    'bar': 150
}
```

---

### 3. **Order Blocks Formed** (`self.order_blocks_formed`)
All order blocks created during trading:
- `type`: 'bullish' or 'bearish'
- `top`: Upper price boundary
- `btm`: Lower price boundary
- `time`: When OB was formed
- `bar`: Bar index

**Example:**
```python
{
    'type': 'bullish',
    'top': 50500.0,
    'btm': 50000.0,
    'time': 1699401600000,
    'bar': 145
}
```

---

### 4. **Candle Log** (`self.candle_log`)
Recent candle data (last 100):
- `time`: Unix timestamp
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

---

### 5. **Performance Metrics**
Real-time calculations:
- `total_trades`: Number of completed trades
- `winning_trades`: Number of profitable trades
- `losing_trades`: Number of losing trades
- `win_rate`: Win percentage
- `initial_capital`: Starting capital
- `current_capital`: Current capital
- `total_return_pct`: Total return percentage

---

## 🔌 Webhook API

Access all tracked data via webhooks. Base URL:
```
https://profitview.net/trading/bot/YOUR_WEBHOOK_SECRET/
```

### 1. **Get Status**
**Endpoint:** `GET /status`

Returns current bot status and performance summary.

**Response:**
```json
{
    "venue": "WooPaper",
    "symbol": "PERP_BTC_USDT",
    "bar_index": 250,
    "bullish_obs": 2,
    "bearish_obs": 1,
    "position": {...},
    "parameters": {...},
    "performance": {
        "total_trades": 10,
        "winning_trades": 7,
        "losing_trades": 3,
        "win_rate": 70.0,
        "initial_capital": 10000,
        "current_capital": 10500,
        "total_return_pct": 5.0
    }
}
```

---

### 2. **Get Trade History**
**Endpoint:** `GET /trade_history?limit=50`

Returns completed trades (default: last 50).

**Parameters:**
- `limit` (optional): Number of trades to return

**Response:**
```json
{
    "trades": [...],
    "total_count": 25
}
```

---

### 3. **Get Equity Curve**
**Endpoint:** `GET /equity_curve?limit=100`

Returns equity tracking data.

**Parameters:**
- `limit` (optional): Number of data points

**Response:**
```json
{
    "equity_curve": [...],
    "total_points": 500
}
```

---

### 4. **Get Order Blocks**
**Endpoint:** `GET /order_blocks?limit=20`

Returns current and historical order blocks.

**Response:**
```json
{
    "current_bullish": [...],
    "current_bearish": [...],
    "history": [...],
    "total_formed": 45
}
```

---

### 5. **Get Performance**
**Endpoint:** `GET /performance`

Detailed performance analytics.

**Response:**
```json
{
    "total_trades": 25,
    "winning_trades": 16,
    "losing_trades": 9,
    "win_rate": 64.0,
    "avg_win": 2.5,
    "avg_loss": -1.2,
    "largest_win": 5.5,
    "largest_loss": -2.0,
    "profit_factor": 1.85,
    "max_drawdown": 3.2,
    "total_return_pct": 8.5,
    "initial_capital": 10000,
    "current_capital": 10850
}
```

---

### 6. **Get Recent Candles**
**Endpoint:** `GET /recent_candles?limit=20`

Returns recent candle data.

**Response:**
```json
{
    "candles": [...],
    "total_tracked": 100
}
```

---

## 📝 Logging

All events are logged via ProfitView's `logger` with clear formatting:

### Entry Signals
```
============================================================
🟢 LONG SIGNAL
   Entry: $50000.00
   Stop Loss: $49000.00
   Take Profit: $52000.00
============================================================
```

### Exit Signals
```
============================================================
⚪ CLOSE SIGNAL - Reason: TP
   Entry: $50000.00
   Exit: $52000.00
   P&L: +2.00%
   Bars Held: 12
============================================================
💰 Capital Update: $10,200.00 (+2.00%)
📊 Performance: 5 trades, 80.0% win rate
```

### Order Blocks
```
📦 Bullish Order Block formed: 50000.00 - 50500.00
📊 State: 2 bull OBs, 1 bear OBs
```

### Order Updates
```
📝 Order Update: WooPaper PERP_BTC_USDT Buy @ 50000
   Status: filled, Remaining: 0
✅ Fill: WooPaper PERP_BTC_USDT Buy 0.01 @ 50000.00
```

---

## 💡 Usage in Jupyter

You can fetch and analyze this data from Jupyter:

```python
import requests

# Your webhook URL
webhook_url = "https://profitview.net/trading/bot/YOUR_SECRET"

# Get current status
status = requests.get(f"{webhook_url}/status").json()
print(f"Win Rate: {status['performance']['win_rate']:.1f}%")

# Get trade history
trades = requests.get(f"{webhook_url}/trade_history?limit=20").json()
for trade in trades['trades']:
    print(f"{trade['side']}: {trade['pnl_pct']:+.2f}%")

# Get performance metrics
perf = requests.get(f"{webhook_url}/performance").json()
print(f"Profit Factor: {perf['profit_factor']:.2f}")
print(f"Max Drawdown: {perf['max_drawdown']:.2f}%")

# Get equity curve for plotting
equity = requests.get(f"{webhook_url}/equity_curve?limit=500").json()
# Plot with matplotlib...
```

---

## 🎯 Benefits

1. **Complete Transparency**: Every trade and decision is logged
2. **Real-time Monitoring**: Check bot performance anytime via webhooks
3. **Performance Analysis**: Calculate metrics like Sharpe ratio, profit factor
4. **Debugging**: Track order blocks, candles, and strategy state
5. **Visualization**: Export data to create charts in Jupyter
6. **Auditing**: Full trade history for tax/compliance purposes

---

## 📊 Data Retention

- **Trade History**: Unlimited (all trades stored)
- **Equity Curve**: Last 1000 data points
- **Candle Log**: Last 100 candles
- **Order Blocks History**: Unlimited

Note: Arrays are trimmed to prevent excessive memory usage in long-running bots.

---

## 🔒 Security

All webhooks require your ProfitView webhook secret. Never share this secret publicly!

Example URL structure:
```
https://profitview.net/trading/bot/Ff9LhSRKNo3CVQ/status
                                     ^^^^^^^^^^^^^^^^^
                                     Your secret key
```

---

## ✅ Ready to Use

All tracking is **automatically enabled** when you deploy `profitview_OB.py` to ProfitView. No additional setup required!

Just deploy and start trading - all data will be logged and accessible via webhooks.
