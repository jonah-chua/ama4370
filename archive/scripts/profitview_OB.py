"""
===========================================
PROFITVIEW ORDER BLOCKS STRATEGY - MODULAR
===========================================
Clean separation of concerns:
1. OrderBlockStrategy: Pure strategy logic (no ProfitView dependencies)
2. Trading: ProfitView live trading bot (ready to deploy)

This file is READY FOR DEPLOYMENT to ProfitView!
"""

from collections import deque


# =============================================================================
# CORE STRATEGY CLASS (Platform-independent)
# =============================================================================

class OrderBlockStrategy:
    """
    Dynamic Order Blocks Strategy Logic
    
    This class contains ONLY strategy logic - no backtesting, no live trading.
    Can be used by both ProfitView live trading and Jupyter backtesting.
    """
    
    def __init__(self, params=None):
        """
        Initialize strategy with parameters
        
        Args:
            params: dict with strategy parameters
                - swing_lookback: int (default 10)
                - use_body: bool (default True)
                - sweep_confirmation: bool (default True)
                - max_ob_age_bars: int (default 50)
                - stop_loss_pct: float (default 2.0)
                - take_profit_pct: float (default 4.0)
                - risk_reward_ratio: float (default 2.0)
                - use_trailing_stop: bool (default False)
                - trailing_stop_activation: float (default 1.0)
                - trailing_stop_distance: float (default 0.5)
        """
        # Default parameters
        self.params = {
            'swing_lookback': 10,
            'use_body': True,
            'sweep_confirmation': True,
            'max_ob_age_bars': 50,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'risk_reward_ratio': 2.0,
            'use_trailing_stop': False,
            'trailing_stop_activation': 1.0,
            'trailing_stop_distance': 0.5,
        }
        
        # Override with provided parameters
        if params:
            self.params.update(params)
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset strategy state"""
        self.candle_history = deque(maxlen=100)
        self.bullish_obs = []
        self.bearish_obs = []
        self.swing_top = {'price': None, 'bar': None, 'crossed': False}
        self.swing_btm = {'price': None, 'bar': None, 'crossed': False}
        self.bar_index = 0
        self.current_position = None
    
    def on_candle_close(self, candle):
        """
        Main strategy logic - called for each candle close
        
        Args:
            candle: dict with keys: open, high, low, close, volume, time
            
        Returns:
            dict with signals: {'action': None/'long'/'short'/'close', 'price': float, ...}
        """
        self.bar_index += 1
        self.candle_history.append(candle)
        
        if len(self.candle_history) < self.params['swing_lookback'] + 1:
            return {'action': None}
        
        # Update strategy state
        self.detect_swings()
        self.identify_order_blocks(candle)
        self.cleanup_order_blocks(candle)
        
        # Check for entry signals
        entry_signal = self.check_sweep_entry(candle)
        if entry_signal:
            return entry_signal
        
        # Check for exit signals
        exit_signal = self.manage_position(candle)
        if exit_signal:
            return exit_signal
        
        return {'action': None}
    
    def detect_swings(self):
        """Detect swing highs and lows"""
        candles = list(self.candle_history)
        length = self.params['swing_lookback']
        
        if len(candles) < length * 2:
            return
        
        highs = [c['high'] for c in candles[-(length*2):]]
        lows = [c['low'] for c in candles[-(length*2):]]
        
        # Check for swing high
        pivot_high = highs[length]
        is_swing_high = all(pivot_high >= h for h in highs[:length]) and \
                        all(pivot_high >= h for h in highs[length+1:length*2])
        
        # Check for swing low
        pivot_low = lows[length]
        is_swing_low = all(pivot_low <= l for l in lows[:length]) and \
                       all(pivot_low <= l for l in lows[length+1:length*2])
        
        if is_swing_high:
            self.swing_top = {'price': pivot_high, 'bar': self.bar_index - length, 'crossed': False}
        
        if is_swing_low:
            self.swing_btm = {'price': pivot_low, 'bar': self.bar_index - length, 'crossed': False}
    
    def identify_order_blocks(self, current_candle):
        """Identify order blocks when price crosses swings"""
        close = current_candle['close']
        
        # Bullish order block (price crosses above swing high)
        if self.swing_top['price'] and not self.swing_top['crossed']:
            if close > self.swing_top['price']:
                self.swing_top['crossed'] = True
                ob = self.find_bullish_ob(self.swing_top['bar'])
                if ob:
                    ob['time'] = current_candle.get('time', current_candle.get('timestamp', 0))
                    self.bullish_obs.append(ob)
        
        # Bearish order block (price crosses below swing low)
        if self.swing_btm['price'] and not self.swing_btm['crossed']:
            if close < self.swing_btm['price']:
                self.swing_btm['crossed'] = True
                ob = self.find_bearish_ob(self.swing_btm['bar'])
                if ob:
                    ob['time'] = current_candle.get('time', current_candle.get('timestamp', 0))
                    self.bearish_obs.append(ob)
    
    def find_bullish_ob(self, swing_bar):
        """Find bullish order block before breakout"""
        candles = list(self.candle_history)
        minima = float('inf')
        maxima = None
        
        for c in candles[-(self.bar_index - swing_bar):]:
            c_min = min(c['close'], c['open']) if self.params['use_body'] else c['low']
            c_max = max(c['close'], c['open']) if self.params['use_body'] else c['high']
            
            if c_min < minima:
                minima = c_min
                maxima = c_max
        
        if maxima and minima < float('inf'):
            return {'top': maxima, 'btm': minima, 'bar': self.bar_index, 'type': 'bull'}
        return None
    
    def find_bearish_ob(self, swing_bar):
        """Find bearish order block before breakdown"""
        candles = list(self.candle_history)
        maxima = float('-inf')
        minima = None
        
        for c in candles[-(self.bar_index - swing_bar):]:
            c_max = max(c['close'], c['open']) if self.params['use_body'] else c['high']
            c_min = min(c['close'], c['open']) if self.params['use_body'] else c['low']
            
            if c_max > maxima:
                maxima = c_max
                minima = c_min
        
        if minima and maxima > float('-inf'):
            return {'top': maxima, 'btm': minima, 'bar': self.bar_index, 'type': 'bear'}
        return None
    
    def cleanup_order_blocks(self, candle):
        """Remove broken or expired order blocks"""
        close = candle['close']
        open_price = candle['open']
        max_age = self.params['max_ob_age_bars']
        
        # Remove bullish OBs that are broken or too old
        self.bullish_obs = [
            ob for ob in self.bullish_obs
            if min(close, open_price) >= ob['btm'] and 
               (self.bar_index - ob['bar']) <= max_age
        ]
        
        # Remove bearish OBs that are broken or too old
        self.bearish_obs = [
            ob for ob in self.bearish_obs
            if max(close, open_price) <= ob['top'] and
               (self.bar_index - ob['bar']) <= max_age
        ]
    
    def check_sweep_entry(self, candle):
        """
        Check for sweep entry signals - with opposite signal exit support
        
        Returns:
            dict: Signal info with 'action' key
        """
        high, low, close = candle['high'], candle['low'], candle['close']
        
        # Check for OPPOSITE signal to close current position
        if self.current_position:
            pos = self.current_position
            
            # If SHORT, check for bullish sweep to close
            if pos['side'] == 'short':
                for ob in self.bullish_obs:
                    if low < ob['btm'] and close > ob['btm']:
                        if not self.params['sweep_confirmation'] or close < ob['top']:
                            # Close short on bullish signal (using standard close format)
                            return self.close_position('opposite_signal', candle)
            
            # If LONG, check for bearish sweep to close
            elif pos['side'] == 'long':
                for ob in self.bearish_obs:
                    if high > ob['top'] and close < ob['top']:
                        if not self.params['sweep_confirmation'] or close > ob['btm']:
                            # Close long on bearish signal (using standard close format)
                            return self.close_position('opposite_signal', candle)
            
            return {'action': None}
        
        # No position - check for entry signals
        # Check bullish sweeps
        for ob in self.bullish_obs:
            if low < ob['btm'] and close > ob['btm']:
                if not self.params['sweep_confirmation'] or close < ob['top']:
                    # Calculate entry
                    entry = candle['close']
                    # Stop loss BELOW entry by stop_loss_pct
                    sl = entry * (1 - self.params['stop_loss_pct'] / 100)
                    # Take profit based on risk:reward ratio
                    tp = entry + (entry - sl) * self.params['risk_reward_ratio']
                    
                    # ENTER POSITION DIRECTLY
                    self.current_position = {
                        'side': 'long',
                        'entry': entry,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'entry_bar': self.bar_index,
                        'entry_time': candle.get('time', candle.get('timestamp', 0)),
                        'highest_price': entry  # For trailing stop
                    }
                    
                    return {'action': 'long', 'entry': entry, 'stop_loss': sl, 'take_profit': tp}
        
        # Check bearish sweeps
        for ob in self.bearish_obs:
            if high > ob['top'] and close < ob['top']:
                if not self.params['sweep_confirmation'] or close > ob['btm']:
                    # Calculate entry
                    entry = candle['close']
                    # Stop loss ABOVE entry by stop_loss_pct
                    sl = entry * (1 + self.params['stop_loss_pct'] / 100)
                    # Take profit based on risk:reward ratio
                    tp = entry - (sl - entry) * self.params['risk_reward_ratio']
                    
                    # ENTER POSITION DIRECTLY
                    self.current_position = {
                        'side': 'short',
                        'entry': entry,
                        'stop_loss': sl,
                        'take_profit': tp,
                        'entry_bar': self.bar_index,
                        'entry_time': candle.get('time', candle.get('timestamp', 0)),
                        'lowest_price': entry  # For trailing stop
                    }
                    
                    return {'action': 'short', 'entry': entry, 'stop_loss': sl, 'take_profit': tp}
        
        return {'action': None}
    
    def manage_position(self, candle):
        """
        Manage open position (stop loss, take profit, trailing stop)
        
        Returns:
            dict: {'action': 'close', 'reason': 'SL'/'TP', 'price': float}
            or {'action': None}
        """
        if not self.current_position:
            return {'action': None}
        
        close = candle['close']
        pos = self.current_position
        
        # Update trailing stop if enabled
        if self.params['use_trailing_stop']:
            if pos['side'] == 'long':
                # Update highest price
                if close > pos['highest_price']:
                    pos['highest_price'] = close
                
                # Check if we should activate trailing stop
                profit_pct = ((close - pos['entry']) / pos['entry']) * 100
                if profit_pct >= self.params['trailing_stop_activation']:
                    # Calculate trailing stop
                    trailing_sl = pos['highest_price'] * (1 - self.params['trailing_stop_distance'] / 100)
                    # Use trailing stop if it's higher than original stop
                    if trailing_sl > pos['stop_loss']:
                        pos['stop_loss'] = trailing_sl
            
            else:  # short
                # Update lowest price
                if close < pos['lowest_price']:
                    pos['lowest_price'] = close
                
                # Check if we should activate trailing stop
                profit_pct = ((pos['entry'] - close) / pos['entry']) * 100
                if profit_pct >= self.params['trailing_stop_activation']:
                    # Calculate trailing stop
                    trailing_sl = pos['lowest_price'] * (1 + self.params['trailing_stop_distance'] / 100)
                    # Use trailing stop if it's lower than original stop
                    if trailing_sl < pos['stop_loss']:
                        pos['stop_loss'] = trailing_sl
        
        # Check exit conditions
        if pos['side'] == 'long':
            if close <= pos['stop_loss']:
                return self.close_position('SL', candle)
            elif close >= pos['take_profit']:
                return self.close_position('TP', candle)
        else:  # short
            if close >= pos['stop_loss']:
                return self.close_position('SL', candle)
            elif close <= pos['take_profit']:
                return self.close_position('TP', candle)
        
        return {'action': None}
    
    def close_position(self, reason, candle):
        """Close current position"""
        if not self.current_position:
            return {'action': None}
        
        pos = self.current_position
        exit_price = candle['close']
        
        # Calculate P&L
        if pos['side'] == 'long':
            pnl_pct = ((exit_price - pos['entry']) / pos['entry']) * 100
        else:
            pnl_pct = ((pos['entry'] - exit_price) / pos['entry']) * 100
        
        signal = {
            'action': 'close',
            'side': pos['side'],
            'entry': pos['entry'],
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': candle['time'],
            'bars_held': self.bar_index - pos['entry_bar']
        }
        
        self.current_position = None
        return signal


# =============================================================================
# PROFITVIEW LIVE TRADING BOT (Ready to Deploy!)
# =============================================================================

try:
    from profitview import Link, http, logger
    
    class Trading(Link):
        """
        ProfitView Live Trading Bot using Order Blocks Strategy
        
        DEPLOYMENT READY!
        - No backtesting code
        - Clean ProfitView integration
        - Uses modular strategy class
        - Configurable parameters
        """
        
        # ============================================
        # CONFIGURATION
        # ============================================
        SRC = 'woo'
        VENUE = 'WooPaper'  # Change to 'Woo' for live trading
        SYMBOL = 'PERP_BTC_USDT'
        
        def __init__(self):
            """Initialize ProfitView trading bot"""
            super().__init__()
            
            logger.info("=" * 60)
            logger.info("🚀 Initializing Order Blocks Live Trading Bot")
            logger.info("=" * 60)
            
            # Strategy parameters (TUNE THESE)
            strategy_params = {
                'swing_lookback': 10,
                'use_body': True,
                'sweep_confirmation': True,
                'max_ob_age_bars': 50,
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'risk_reward_ratio': 2.0,
                'use_trailing_stop': True,  # Enable trailing stop!
                'trailing_stop_activation': 1.0,  # Activate after 1% profit
                'trailing_stop_distance': 0.5,   # Trail 0.5% behind peak
            }
            
            # Position sizing
            self.position_size = 0.01  # Size in BTC (adjust for your capital)
            
            # Initialize strategy
            self.strategy = OrderBlockStrategy(strategy_params)
            
            # ============================================
            # TRACKING & LOGGING (for analytics)
            # ============================================
            self.trade_history = []  # All completed trades
            self.equity_curve = []   # Track capital over time
            self.order_blocks_formed = []  # All order blocks created
            self.candle_log = []     # Recent candle data
            self.initial_capital = 10000  # Starting capital
            self.current_capital = self.initial_capital
            
            # Performance metrics
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            
            logger.info("✅ Strategy initialized with parameters:")
            for key, value in strategy_params.items():
                logger.info(f"   {key}: {value}")
            logger.info(f"   position_size: {self.position_size} BTC")
            logger.info(f"   initial_capital: ${self.initial_capital:,.2f}")
            logger.info("✅ Tracking enabled: trades, equity, order blocks")
            logger.info("=" * 60)
        
        # ============================================
        # LIVE TRADING CALLBACKS
        # ============================================
        
        def quote_update(self, src, sym, data):
            """
            Called on every quote update (top-of-book)
            We'll use this for real-time candle formation
            """
            # You can implement real-time candle tracking here if needed
            pass
        
        def order_update(self, src, sym, data):
            """Called when order status changes"""
            logger.info(f"📝 Order Update: {data['venue']} {sym} {data['side']} @ {data['order_price']}")
            logger.info(f"   Status: {data.get('status', 'unknown')}, Remaining: {data['remain_size']}")
        
        def fill_update(self, src, sym, data):
            """Called when order is filled"""
            logger.info(f"✅ Fill: {data['venue']} {sym} {data['side']} {data['fill_size']} @ {data['fill_price']}")
        
        def position_update(self, src, sym, data):
            """Called when position changes"""
            logger.info(f"📊 Position Update: {data['venue']} {sym}")
            logger.info(f"   Size: {data['pos_size']}, Entry: {data.get('entry_price', 'N/A')}")
            logger.info(f"   Mark Price: {data.get('mark_price', 'N/A')}, Liq Price: {data.get('liq_price', 'N/A')}")
        
        # ============================================
        # STRATEGY EXECUTION (Called periodically)
        # ============================================
        
        def on_candle_close(self, candle):
            """
            Process candle and execute strategy signals
            Call this method when you have a complete candle
            """
            logger.info(f"🕐 Processing candle: {candle['time']} | Close: {candle['close']:.2f}")
            
            # Track candle (keep last 100)
            self.candle_log.append({
                'time': candle['time'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle.get('volume', 0)
            })
            if len(self.candle_log) > 100:
                self.candle_log.pop(0)
            
            # Track equity
            self.equity_curve.append({
                'time': candle['time'],
                'price': candle['close'],
                'capital': self.current_capital,
                'bar': self.strategy.bar_index
            })
            if len(self.equity_curve) > 1000:
                self.equity_curve.pop(0)
            
            # Track order blocks formed
            prev_bull_count = len(self.strategy.bullish_obs)
            prev_bear_count = len(self.strategy.bearish_obs)
            
            # Get strategy signal
            signal = self.strategy.on_candle_close(candle)
            
            # Log new order blocks
            if len(self.strategy.bullish_obs) > prev_bull_count:
                new_ob = self.strategy.bullish_obs[-1]
                self.order_blocks_formed.append({
                    'type': 'bullish',
                    'top': new_ob['top'],
                    'btm': new_ob['btm'],
                    'time': candle['time'],
                    'bar': self.strategy.bar_index
                })
                logger.info(f"📦 Bullish Order Block formed: {new_ob['btm']:.2f} - {new_ob['top']:.2f}")
            
            if len(self.strategy.bearish_obs) > prev_bear_count:
                new_ob = self.strategy.bearish_obs[-1]
                self.order_blocks_formed.append({
                    'type': 'bearish',
                    'top': new_ob['top'],
                    'btm': new_ob['btm'],
                    'time': candle['time'],
                    'bar': self.strategy.bar_index
                })
                logger.info(f"📦 Bearish Order Block formed: {new_ob['btm']:.2f} - {new_ob['top']:.2f}")
            
            # Log strategy state
            logger.info(f"📊 State: {len(self.strategy.bullish_obs)} bull OBs, {len(self.strategy.bearish_obs)} bear OBs")
            
            # Execute signal
            if signal['action'] == 'long':
                self.execute_long(signal)
            elif signal['action'] == 'short':
                self.execute_short(signal)
            elif signal['action'] == 'close':
                self.execute_close(signal)
        
        def execute_long(self, signal):
            """Execute long entry"""
            logger.info("=" * 60)
            logger.info("🟢 LONG SIGNAL")
            logger.info(f"   Entry: ${signal['entry']:.2f}")
            logger.info(f"   Stop Loss: ${signal['stop_loss']:.2f}")
            logger.info(f"   Take Profit: ${signal['take_profit']:.2f}")
            logger.info("=" * 60)
            
            # Place market order
            result = self.create_market_order(
                venue=self.VENUE,
                sym=self.SYMBOL,
                side='Buy',
                size=self.position_size
            )
            
            if result['error']:
                logger.error(f"❌ Long order failed: {result['error']}")
            else:
                logger.info(f"✅ Long order placed: {result['data']}")
        
        def execute_short(self, signal):
            """Execute short entry"""
            logger.info("=" * 60)
            logger.info("🔴 SHORT SIGNAL")
            logger.info(f"   Entry: ${signal['entry']:.2f}")
            logger.info(f"   Stop Loss: ${signal['stop_loss']:.2f}")
            logger.info(f"   Take Profit: ${signal['take_profit']:.2f}")
            logger.info("=" * 60)
            
            # Place market order
            result = self.create_market_order(
                venue=self.VENUE,
                sym=self.SYMBOL,
                side='Sell',
                size=self.position_size
            )
            
            if result['error']:
                logger.error(f"❌ Short order failed: {result['error']}")
            else:
                logger.info(f"✅ Short order placed: {result['data']}")
        
        def execute_close(self, signal):
            """Execute position close"""
            logger.info("=" * 60)
            logger.info(f"⚪ CLOSE SIGNAL - Reason: {signal['reason']}")
            logger.info(f"   Entry: ${signal['entry']:.2f}")
            logger.info(f"   Exit: ${signal['exit']:.2f}")
            logger.info(f"   P&L: {signal['pnl_pct']:+.2f}%")
            logger.info(f"   Bars Held: {signal['bars_held']}")
            logger.info("=" * 60)
            
            # Calculate P&L in USD (simulated for tracking)
            position_value = self.current_capital * 0.01  # 1% of capital
            pnl_usd = position_value * (signal['pnl_pct'] / 100)
            self.current_capital += pnl_usd
            
            # Track trade
            trade = {
                'side': signal['side'],
                'entry': signal['entry'],
                'exit': signal['exit'],
                'entry_time': signal['entry_time'],
                'exit_time': signal['exit_time'],
                'pnl_pct': signal['pnl_pct'],
                'pnl_usd': pnl_usd,
                'reason': signal['reason'],
                'bars_held': signal['bars_held'],
                'capital_after': self.current_capital
            }
            self.trade_history.append(trade)
            
            # Update metrics
            self.total_trades += 1
            if signal['pnl_pct'] > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Log performance
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
            
            logger.info(f"💰 Capital Update: ${self.current_capital:,.2f} ({total_return:+.2f}%)")
            logger.info(f"📊 Performance: {self.total_trades} trades, {win_rate:.1f}% win rate")
            
            # Close position (opposite side)
            side_to_close = 'Sell' if signal['side'] == 'long' else 'Buy'
            
            result = self.create_market_order(
                venue=self.VENUE,
                sym=self.SYMBOL,
                side=side_to_close,
                size=self.position_size
            )
            
            if result['error']:
                logger.error(f"❌ Close order failed: {result['error']}")
            else:
                logger.info(f"✅ Position closed: {result['data']}")
        
        # ============================================
        # WEBHOOKS (Optional - for monitoring)
        # ============================================
        
        @http.route
        def get_status(self, data):
            """Get current strategy status"""
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital * 100)
            
            return {
                'venue': self.VENUE,
                'symbol': self.SYMBOL,
                'bar_index': self.strategy.bar_index,
                'bullish_obs': len(self.strategy.bullish_obs),
                'bearish_obs': len(self.strategy.bearish_obs),
                'position': self.strategy.current_position,
                'parameters': self.strategy.params,
                'performance': {
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': win_rate,
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'total_return_pct': total_return
                }
            }
        
        @http.route
        def get_trade_history(self, data):
            """Get all completed trades"""
            limit = data.get('limit', 50)  # Default last 50 trades
            return {
                'trades': self.trade_history[-limit:],
                'total_count': len(self.trade_history)
            }
        
        @http.route
        def get_equity_curve(self, data):
            """Get equity curve data"""
            limit = data.get('limit', 100)  # Default last 100 points
            return {
                'equity_curve': self.equity_curve[-limit:],
                'total_points': len(self.equity_curve)
            }
        
        @http.route
        def get_order_blocks(self, data):
            """Get all order blocks formed"""
            limit = data.get('limit', 20)  # Default last 20 OBs
            return {
                'current_bullish': [
                    {'top': ob['top'], 'btm': ob['btm'], 'bar': ob['bar']}
                    for ob in self.strategy.bullish_obs
                ],
                'current_bearish': [
                    {'top': ob['top'], 'btm': ob['btm'], 'bar': ob['bar']}
                    for ob in self.strategy.bearish_obs
                ],
                'history': self.order_blocks_formed[-limit:],
                'total_formed': len(self.order_blocks_formed)
            }
        
        @http.route
        def get_performance(self, data):
            """Get detailed performance metrics"""
            if not self.trade_history:
                return {'error': 'No trades yet'}
            
            import numpy as np
            
            trades = self.trade_history
            pnls = [t['pnl_pct'] for t in trades]
            wins = [t for t in trades if t['pnl_pct'] > 0]
            losses = [t for t in trades if t['pnl_pct'] < 0]
            
            win_rate = (len(wins) / len(trades) * 100) if trades else 0
            avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
            
            # Calculate profit factor
            win_usd = sum([t['pnl_usd'] for t in wins]) if wins else 0
            loss_usd = sum([t['pnl_usd'] for t in losses]) if losses else 0
            profit_factor = abs(win_usd / loss_usd) if loss_usd != 0 else 0
            
            # Calculate max drawdown
            peak = self.initial_capital
            max_dd = 0
            for point in self.equity_curve:
                if point['capital'] > peak:
                    peak = point['capital']
                dd = ((peak - point['capital']) / peak) * 100
                max_dd = max(max_dd, dd)
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': max(pnls) if pnls else 0,
                'largest_loss': min(pnls) if pnls else 0,
                'profit_factor': profit_factor,
                'max_drawdown': max_dd,
                'total_return_pct': ((self.current_capital - self.initial_capital) / self.initial_capital * 100),
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital
            }
        
        @http.route
        def get_recent_candles(self, data):
            """Get recent candle data"""
            limit = data.get('limit', 20)
            return {
                'candles': self.candle_log[-limit:],
                'total_tracked': len(self.candle_log)
            }

except ImportError:
    # ProfitView not available (we're in Jupyter)
    pass
