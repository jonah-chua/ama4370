# Accurate PnL Tracking - Hybrid Approach

## Overview
Implemented **Option C: Hybrid approach** for more accurate PnL calculation using actual fill prices and fees from exchange when available, with fallback to estimates.

---

## What Changed

### 1. New Position Tracking Fields
Added to `TradingPosition` dataclass:
- `actual_entry_price`: Actual fill price from `fill_update()` callback
- `actual_exit_price`: Actual exit fill price from exchange
- `entry_fees_paid`: Sum of actual fees paid on entry fills
- `exit_fees_paid`: Sum of actual fees paid on exit fills  
- `is_closing`: Flag to identify exit fills vs entry fills

### 2. Enhanced `fill_update()` Callback
Now tracks:
- **Entry fills**: Captures actual fill price and fees for positions in `pending_orders`
- **Exit fills**: Captures actual exit fill price and fees for positions being closed
- **Fee accumulation**: Sums fees across multiple partial fills
- **Detailed logging**: Shows individual fill prices and cumulative fees

```python
# Entry Fill Example Log:
Entry Fill: Buy 0.5 @ 2451.23 | Fee: 0.1225 | Total Entry Fees: 0.1225

# Exit Fill Example Log:
Exit Fill: Sell 0.5 @ 2458.67 | Fee: 0.1229 | Total Exit Fees: 0.1229
```

### 3. Improved `close_position()` Logic
**Price Selection**:
- ✅ Uses `actual_exit_price` if available from fills
- ⚠️ Falls back to `current_price` estimate if no fill data

**Fee Calculation**:
- ✅ Uses `entry_fees_paid + exit_fees_paid` if both available
- ⚠️ Falls back to `COMMISSION_PERCENT + SLIPPAGE_PERCENT` estimates if no fill data

**Logging Enhancement**:
```python
# Shows whether actual or estimated prices were used:
✓ Position closed: PnL $7.23 (0.29%) | Exit price: actual
✓ Position closed: PnL $5.81 (0.24%) | Exit price: estimated
```

### 4. Enhanced HTTP Endpoints
All position data now includes:
- `actual_entry_price` - real fill price or null
- `actual_exit_price` - real exit fill price or null  
- `entry_fees_paid` - actual fees paid (sum of fills)
- `exit_fees_paid` - actual exit fees paid
- `total_fees` - combined fees (entry + exit)
- `used_actual_prices` - boolean flag indicating if PnL used real prices

---

## Benefits

✅ **Most Accurate PnL**: Uses real exchange data when available  
✅ **Real Fee Tracking**: No more estimates - see actual fees paid  
✅ **Fallback Safety**: Still works if fill callbacks fail  
✅ **Transparency**: Logs show whether actual or estimated prices were used  
✅ **Multi-Algo Safe**: Each position tracks its own fills independently  

---

## Answering Your Questions

### Q1: Can we calculate accurate PnL from entry/exit + units?
**A: Yes, now we do!**
- We capture actual fill prices from `fill_update()` 
- Formula: `PnL = (exit_price - entry_price) * position_size - total_fees`
- Fees are actual exchange fees, not estimates
- If another algo is running, each position tracks its own fills via `order_id`

### Q2: Is slippage needed if we know actual entry price?
**A: No, it's only a fallback now!**
- If `actual_entry_price` exists → use real price (slippage already in it)
- If `actual_entry_price` is None → estimate with `entry_price + slippage`
- Same for exit prices
- Most of the time you'll get actual prices, estimates are rare edge cases

### Q3: Are we saving all trade actions?
**A: Yes! ✅**

Saved per trade:
- Entry/exit timestamps
- Actual vs estimated prices (both tracked)
- All fees paid (broken down by entry/exit)
- OB details, SL/TP, position size
- Trailing stop activation, peak prices
- Exit reason (SL/TP/trailing/holding_period)
- PnL (percent and dollars)

Accessible via:
- `GET /positions` - returns all open + closed positions with full details
- `GET /status` - returns cumulative stats
- Each closed position shows `used_actual_prices: true/false`

---

## Example API Response

```json
{
  "closed_positions": [
    {
      "type": "long",
      "entry_price": 2450.00,
      "actual_entry_price": 2451.23,
      "entry_fees_paid": 0.1226,
      "exit_price": 2458.50,
      "actual_exit_price": 2458.67,
      "exit_fees_paid": 0.1229,
      "pnl_dollars": 7.23,
      "total_fees": 0.2455,
      "used_actual_prices": true,
      "exit_reason": "take_profit"
    }
  ]
}
```

---

## Testing Checklist

When bot is live:
1. ✅ Check logs for "Entry Fill: ..." messages with actual prices
2. ✅ Check logs for "Exit Fill: ..." messages with actual fees
3. ✅ Verify closed position log shows "Exit price: actual" not "estimated"
4. ✅ Call `GET /positions` and verify `used_actual_prices: true`
5. ✅ Compare `entry_price` vs `actual_entry_price` (should differ slightly due to slippage)
6. ✅ Verify `total_fees` matches sum of `entry_fees_paid + exit_fees_paid`

---

## Edge Cases Handled

✅ **Partial fills**: Accumulates fees, uses latest fill price (close enough for market orders)  
✅ **Missing fill data**: Falls back to estimated prices + commission/slippage  
✅ **Order failure**: Resets `is_closing` flag so position isn't stuck  
✅ **Multiple algos**: Each position's `order_id` links fills correctly  

---

## Commission Note

`COMMISSION_PERCENT` (0.1%) is still used as:
1. **Fallback estimate** if no fill data received
2. **WooPaper default** - update via `/update_params` if your tier differs

You can check your actual tier fees at: https://support.woo.org/hc/en-001/articles/4410275602585

Update if needed:
```bash
curl -X POST http://your-bot-url/update_params \
  -H "Content-Type: application/json" \
  -d '{"commission_percent": 0.05}'  # Example: VIP tier
```
