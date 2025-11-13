# Update Params Endpoint - Quick Reference

## Endpoint
```
POST /update_params
```

## Purpose
Update trading bot parameters at runtime without restarting the strategy.

## Request Format
Send JSON body with parameter names and new values:

```json
{
    "parameter_name": new_value
}
```

## Supported Parameters

### Risk & Capital Management
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `risk_per_trade_percent` | float | 0.1 - 10.0 | 1.5 | % of capital to risk per trade |
| `max_concurrent_positions` | int | 1 - 20 | 3 | Maximum open positions at once |
| `max_position_size_usd` | float | 100 - 1000000 | 100000 | Maximum position value in USD |
| `max_loss_limit` | float | 100 - 100000 | 5000 | Maximum cumulative loss before pause |

### Position Management
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `stop_loss_multiplier` | float | 0.5 - 5.0 | 1.2 | SL = OB_size × this |
| `take_profit_multiplier` | float | 1.0 - 50.0 | 15.0 | TP = SL_distance × this |
| `holding_period_bars` | int | 0 - 2000 | 480 | Close after N candles (0=disabled) |

### Trailing Stops
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `trailing_stop_activation` | float | 1.0 - 10.0 | 2.0 | Activate when profit = this × SL |
| `trailing_stop_percent` | float | 0.1 - 10.0 | 1.0 | Trail by this % from peak |
| `trailing_stop_buffer_candles` | int | 0 - 100 | 10 | Wait N candles after activation |
| `trailing_stop_update_threshold` | float | 0.0 - 5.0 | 0.4 | Update only if price moves this % |

### Order Block Detection
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `swing_length` | int | 5 - 100 | 30 | Pivot detection period |
| `min_strength_ratio` | float | 0.1 - 0.9 | 0.3 | Min volume ratio for signal |
| `break_atr_mult` | float | 0.0 - 1.0 | 0.02 | Breakout must exceed ATR × this |
| `ob_min_sl_atr_mult` | float | 0.0 - 10.0 | 2.0 | Minimum SL = ATR × this |
| `ob_min_sl_pct` | float | 0.0 - 0.1 | 0.01 | Minimum SL = entry × this |

### Entry Filters
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `entry_diff_long_pct` | float | 0.0 - 0.2 | 0.02 | Skip long if price > EMA by this |
| `entry_diff_short_pct` | float | 0.0 - 0.2 | 0.03 | Skip short if price < EMA by this |
| `atr_sideways_threshold_pct` | float | 0.0 - 0.01 | 0.002 | ATR/price threshold for sideways |
| `sideways_leverage_mult` | float | 0.1 - 10.0 | 0.5 | Capital multiplier in sideways |

## Example Requests

### Increase Risk and Position Limits
```bash
curl -X POST https://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{
    "risk_per_trade_percent": 2.5,
    "max_concurrent_positions": 5,
    "max_position_size_usd": 150000
  }'
```

### Adjust Trailing Stop Parameters
```bash
curl -X POST https://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{
    "trailing_stop_activation": 3.0,
    "trailing_stop_percent": 1.5,
    "trailing_stop_buffer_candles": 15
  }'
```

### Tighten Stop Loss, Extend Take Profit
```bash
curl -X POST https://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{
    "stop_loss_multiplier": 1.0,
    "take_profit_multiplier": 20.0
  }'
```

### Update Order Block Detection Sensitivity
```bash
curl -X POST https://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{
    "swing_length": 35,
    "min_strength_ratio": 0.4,
    "break_atr_mult": 0.03
  }'
```

### Increase Max Loss Limit (and reset flag)
```bash
curl -X POST https://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{
    "max_loss_limit": 10000
  }'
```

## Response Format

### Success Response
```json
{
    "status": "success",
    "updated": [
        "risk_per_trade_percent: 1.5 → 2.5",
        "max_concurrent_positions: 3 → 5",
        "max_position_size_usd: 100000.0 → 150000.0"
    ],
    "errors": [],
    "message": "Updated 3 parameter(s)"
}
```

### Error Response
```json
{
    "status": "error",
    "updated": [],
    "errors": [
        "risk_per_trade_percent: value 15.0 out of range [0.1, 10.0]",
        "unknown_param: Unknown parameter"
    ],
    "message": "No valid parameters updated"
}
```

### Partial Success
```json
{
    "status": "success",
    "updated": [
        "swing_length: 30 → 40",
        "  → ob_search_window: 12",
        "  → atr_period: 48",
        "  → ema_period: 16"
    ],
    "errors": [
        "risk_per_trade_percent: value 20.0 out of range [0.1, 10.0]"
    ],
    "message": "Updated 1 parameter(s)"
}
```

## Special Behaviors

### Swing Length Updates
Updating `swing_length` automatically recalculates:
- `ob_search_window` = swing_length × 0.3
- `atr_period` = swing_length × 1.2
- `ema_period` = swing_length × 0.4
- `atr_sideways_window` = swing_length × 0.3

### Max Loss Limit Updates
Updating `max_loss_limit` resets the `max_loss_reached` flag, allowing trading to resume if previously paused.

## Validation

All parameters are validated for:
1. **Type** - Must be correct type (int or float)
2. **Range** - Must be within defined min/max bounds
3. **Existence** - Parameter name must be recognized

Invalid updates are rejected with descriptive error messages.

## Notes

⚠️ **Parameter updates take effect immediately** for new signals. Open positions use the parameters they were opened with.

⚠️ **Use caution** when updating risk parameters during active trading.

⚠️ **Check logs** after updates to confirm changes were applied correctly.

💡 **Best Practice:** Test parameter changes with small adjustments before making large changes.

## Related Endpoints

- `GET /status` - View current parameters and their values
- `POST /pause` - Pause trading before making parameter changes
- `POST /resume` - Resume trading after parameter changes
