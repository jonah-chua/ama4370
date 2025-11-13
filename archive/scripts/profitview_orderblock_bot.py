from profitview import Link, http, logger
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import deque
import numpy as np
import pandas as pd
import threading


SRC = 'woo'                 # Crypto Exchange
VENUE = 'WooPaper'          # Trading Venue (Account)
SYMBOL = 'PERP_BTC_USDT'    # Crypto Product

# =========================
# Data Structures
# =========================
@dataclass
class Candle:
    time: int  # epoch timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBlock:
    kind: str  # "bullish" or "bearish"
    top: float
    btm: float
    start_time: int
    create_time: int
    violated_time: Optional[int] = None
    bullish_str: float = 0.0
    bearish_str: float = 0.0
    vol: float = 0.0
    active: bool = True


@dataclass
class TradingPosition:
    entry_time: int
    entry_price: float
    position_type: str  # "long" or "short"
    position_size: float
    stop_loss: float
    take_profit: float
    ob_size: float
    sl_distance: float
    tp_distance: float
    capital_at_risk: float
    
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_dollars: Optional[float] = None
    
    # Trailing stop fields
    initial_stop_loss: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    trailing_active: bool = False
    trailing_activation_time: Optional[int] = None
    
    order_id: Optional[str] = None  # ProfitView order ID


# =========================
# Main Trading Class
# =========================
class Trading(Link):
    """
    ProfitView Order Block Trading Bot
    
    Inherits from Link base class and implements event-driven callbacks
    for order blocks detection, signal generation, and position management.
    """
    # Ensure class-level defaults so callbacks can reference these before instance init
    SRC = SRC
    VENUE = VENUE
    SYMBOL = SYMBOL
    # Safety defaults (do NOT hardcode capital here). Use None so we fall back to INITIAL_CAPITAL until fetch populates values.
    initialized = False
    current_bid = None
    current_ask = None
    current_price = None
    current_capital = None
    COMMISSION_PERCENT = None
    # Provide a class-level CANDLE_LEVEL so callbacks that run before __init__ don't fail
    CANDLE_LEVEL = '1h'
    
    def __init__(self):
        """Initialize strategy parameters and data structures"""
        # Initialize parent Link class first
        super().__init__()
        
        # === Strategy Parameters ===
        self.SWING_LENGTH = 10
        self.VIOLATION_TYPE = "Close"  # "Close" or "Wick"
        self.HIDE_OVERLAP = True
        self.MIN_STRENGTH_RATIO = 0.30
        self.MIN_TOTAL_VOLUME = 1e-7
        
        # === Position Management ===
        self.STOP_LOSS_MULTIPLIER = 0.5
        self.TAKE_PROFIT_MULTIPLIER = 2.0
        self.MAX_CONCURRENT_POSITIONS = 2
        self.TRAILING_STOP_ACTIVATION = 1.5
        self.TRAILING_STOP_PERCENT = 3.0
        self.TRAILING_STOP_BUFFER_MS = 3600000  # 1 hour in milliseconds
        self.TRAILING_STOP_UPDATE_THRESHOLD = 0.5
        
        # === Capital Management ===
        self.INITIAL_CAPITAL = 10000.0
        self.RISK_PER_TRADE_PERCENT = 3.0
        self.USE_FIXED_CAPITAL = False
        self.MAX_POSITION_SIZE_USD = 100000.0
        
        # === Fees ===
        self.COMMISSION_PERCENT = 0.01
        self.SLIPPAGE_PERCENT = 0.1
        
        # === Trading Configuration ===
        self.VENUE = "WooPaper"  # Default venue (must match account name)
        self.SYMBOL = "PERP_BTC_USDT"  # Default symbol
        self.CANDLE_LEVEL = "1h"  # Candle timeframe
        
        # === State Management ===
        # current_capital will be populated by fetch_account_info; initialize to INITIAL_CAPITAL at instance level
        # but class-level default is None to avoid hardcoding.
        self.current_capital = self.INITIAL_CAPITAL
        self.candles: deque = deque(maxlen=200)  # Store recent candles
        self.order_blocks: List[OrderBlock] = []
        self.positions: List[TradingPosition] = []
        self.open_positions: List[TradingPosition] = []

        # Pivot tracking
        self.last_pivot_high = None  # {'price': float, 'time': int, 'crossed': bool}
        self.last_pivot_low = None

        # Current market state
        self.current_bid = None
        self.current_ask = None
        self.current_price = None

        # Order tracking
        self.pending_orders: Dict[str, TradingPosition] = {}  # order_id -> position
        # Fee tracking
        self.total_fees_paid = 0.0

        # Initialization flag
        self.initialized = False

        # Deferred capital updates recorded while we don't yet know the live balance
        # This avoids seeding or 'faking' the capital; adjustments are applied once fetch_account_info populates current_capital
        self._deferred_capital_adjustments: List[float] = []
        
        print(f"✓ Trading bot initialized - {self.SYMBOL} on {self.VENUE}")
        print(f"✓ Risk per trade: {self.RISK_PER_TRADE_PERCENT}% | Max positions: {self.MAX_CONCURRENT_POSITIONS}")
        print(f"⏳ Waiting for initial data...")
        logger.info("Trading.__init__ completed. Waiting for market events or initialize_candles()")
        # Start a background thread to fetch account/balance/fee info (best-effort)
        try:
            if hasattr(self, 'create_thread'):
                self.create_thread(self.fetch_account_info)
                logger.info('Started background thread to fetch account info via create_thread')
            else:
                t = threading.Thread(target=self.fetch_account_info, daemon=True)
                t.start()
                logger.info('Started background thread to fetch account info via threading.Thread')
        except Exception:
            logger.exception('Failed to start background thread for fetch_account_info')

    
    
    # =========================
    # Initialization
    # =========================
    def initialize_candles(self):
        """Fetch historical candles to populate initial state"""
        if self.initialized:
            return
        
        try:
            logger.info(f"initialize_candles: fetching historical candles for {self.SYMBOL}")
            logger.debug(f"initialize_candles: params level={self.CANDLE_LEVEL} since={self.epoch_now - (200 * 3600000)}")

            response = self.fetch_candles(
                venue=self.VENUE,
                sym=self.SYMBOL,
                level=self.CANDLE_LEVEL,
                since=self.epoch_now - (200 * 3600000)  # Last 200 hours for 1h candles
            )
            logger.debug(f"initialize_candles: fetch_candles response type={type(response)}")
            if not isinstance(response, dict):
                logger.error("initialize_candles: unexpected response from fetch_candles")
                return

            if response.get('error'):
                logger.error(f"initialize_candles: Error fetching candles: {response['error']}")
                return
            
            candle_data = response['data']
            for c in candle_data:
                candle = Candle(
                    time=c['time'],
                    open=c['open'],
                    high=c['high'],
                    low=c['low'],
                    close=c['close'],
                    volume=c['volume']
                )
                self.candles.append(candle)
            
            logger.info(f"✓ Loaded {len(self.candles)} historical candles")
            
            # Process initial order blocks
            self.detect_order_blocks_batch()
            
            self.initialized = True
            logger.info("initialize_candles: initialization complete; bot ready to trade")
            print(f"✓ Bot ready to trade!")
            
        except Exception as e:
            logger.exception(f"initialize_candles: Exception during initialization: {e}")
            print(f"Error during initialization: {e}")


    def fetch_account_info(self):
        """Best-effort: populate current_capital and COMMISSION_PERCENT from the platform.
        Runs in background thread started from __init__.
        """
        try:
            logger.info("fetch_account_info: attempting to fetch account/balance info")
            resp = None
            if hasattr(self, 'fetch_balances'):
                resp = self.fetch_balances(self.VENUE)
            elif hasattr(self, 'fetch_accounts'):
                resp = self.fetch_accounts()
            elif hasattr(self, 'fetch_account'):
                resp = self.fetch_account(self.VENUE)
            else:
                logger.debug('fetch_account_info: no account fetch API available')
                return

            if not resp:
                logger.debug('fetch_account_info: empty response')
                return

            if isinstance(resp, dict) and resp.get('error'):
                logger.warning(f"fetch_account_info: api error: {resp.get('error')}")
                return

            data = resp.get('data') if isinstance(resp, dict) else resp
            acc = None
            if isinstance(data, list) and data:
                # try to find matching venue account
                for a in data:
                    if a.get('venue') == self.VENUE or a.get('account') == self.VENUE:
                        acc = a
                        break
                if acc is None:
                    acc = data[0]
            elif isinstance(data, dict):
                acc = data

            # Try common USD balance keys
            usd_balance = None
            if isinstance(acc, dict):
                for k in ('usd_balance', 'USD', 'balance_usd', 'total_usd', 'balance'):
                    if k in acc and acc[k] is not None:
                        try:
                            usd_balance = float(acc[k])
                            break
                        except Exception:
                            pass

                # balances nested
                if usd_balance is None and 'balances' in acc and isinstance(acc['balances'], dict):
                    for k in ('USDT', 'USD', 'usd'):
                        if k in acc['balances']:
                            try:
                                usd_balance = float(acc['balances'][k].get('available', acc['balances'][k].get('total', 0)))
                                break
                            except Exception:
                                pass

                if usd_balance is not None:
                    self.current_capital = usd_balance
                    logger.info(f"fetch_account_info: set current_capital = {self.current_capital}")

                    # Apply any deferred adjustments (fees, PnL) that occurred before we knew the balance
                    if getattr(self, '_deferred_capital_adjustments', None):
                        total_adj = sum(self._deferred_capital_adjustments)
                        logger.info(f"fetch_account_info: applying deferred capital adjustments: {total_adj}")
                        try:
                            self.current_capital += total_adj
                        except Exception:
                            logger.exception('fetch_account_info: failed to apply deferred adjustments')
                        self._deferred_capital_adjustments = []

                # Try to detect fee percent
                for k in ('maker_fee_percent', 'taker_fee_percent', 'fee_percent', 'fee'):
                    if k in acc and acc[k] is not None:
                        try:
                            self.COMMISSION_PERCENT = float(acc[k])
                            logger.info(f"fetch_account_info: set COMMISSION_PERCENT = {self.COMMISSION_PERCENT}")
                            break
                        except Exception:
                            pass

            logger.info('fetch_account_info: completed')
        except Exception:
            logger.exception('fetch_account_info: unexpected error')
    
    
    # =========================
    # Private Event Callbacks
    # =========================
    def order_update(self, src: str, sym: str, data: dict):
        """Handle order status updates"""
        order_id = data['order_id']
        remain_size = data['remain_size']
        
        # Check if order is filled
        if remain_size == 0 and order_id in self.pending_orders:
            position = self.pending_orders[order_id]
            position.order_id = order_id
            self.open_positions.append(position)
            del self.pending_orders[order_id]
            
            print(f"✓ Position opened: {position.position_type.upper()} {position.position_size} @ {position.entry_price}")
            
            # Publish position to websocket
            self.publish('position_opened', {
                'type': position.position_type,
                'size': position.position_size,
                'entry_price': position.entry_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'time': self.iso_now
            })
    
    
    def fill_update(self, src: str, sym: str, data: dict):
        """Handle trade fill updates"""
        logger.info(f"fill_update: src={src} sym={sym} data_keys={list(data.keys())}")
        # Record per-fill fee if available
        fee = data.get('fee') or data.get('fee_amount') or data.get('fee_value')
        try:
            if fee is not None:
                fee_amt = float(fee)
                self.total_fees_paid += fee_amt
                # best-effort: deduct fee from current capital (assumes fee in USD)
                if self.current_capital is not None:
                    try:
                        self.current_capital -= fee_amt
                    except Exception:
                        logger.exception('fill_update: failed to deduct fee from current_capital')
                else:
                    # Record deferred negative adjustment to apply when balance is known
                    logger.debug('fill_update: current_capital unknown; deferring fee deduction')
                    if not hasattr(self, '_deferred_capital_adjustments'):
                        self._deferred_capital_adjustments = []
                    self._deferred_capital_adjustments.append(-fee_amt)
                logger.info(f"fill_update: recorded fee={fee_amt} total_fees_paid={self.total_fees_paid} current_capital={self.current_capital}")
        except Exception:
            logger.exception('fill_update: failed to record fee')

        # Keep a compact print for quick visibility
        print(f"Fill: {data.get('side')} {data.get('fill_size')} @ {data.get('fill_price')}")
    
    
    def position_update(self, src: str, sym: str, data: dict):
        """Handle position updates from exchange"""
        # This is called for exchange-level position updates
        # We manage positions internally, but can use this for reconciliation
        pass
    
    
    # =========================
    # Public Event Callbacks
    # =========================
    def quote_update(self, src: str, sym: str, data: dict):
        """Handle top-of-book quote updates"""
        logger.info(f"quote_update: src={src} sym={sym}")
        if sym != self.SYMBOL:
            logger.debug("quote_update: symbol mismatch, ignoring")
            return
        
        self.current_bid, _ = data['bid']
        self.current_ask, _ = data['ask']
        self.current_price = (self.current_bid + self.current_ask) / 2
        
        # Only check trailing stops if initialized
        if self.initialized:
            logger.debug("quote_update: checking trailing stops")
            self.check_trailing_stops()
    
    
    def trade_update(self, src: str, sym: str, data: dict):
        """Handle market trade updates"""
        logger.info(f"trade_update: src={src} sym={sym} price={data.get('price')}")
        if sym != self.SYMBOL:
            logger.debug("trade_update: symbol mismatch, ignoring")
            return
        
        # Initialize on first trade
        if not self.initialized:
            logger.info("trade_update: first trade received, calling initialize_candles()")
            self.initialize_candles()
            if not self.initialized:
                logger.warning("trade_update: initialization failed or incomplete; skipping this trade")
                return  # Skip if initialization failed
        
        # Update current price
        self.current_price = data['price']
        
        # Check if we should update candles
        logger.debug("trade_update: updating candles")
        self.update_candles()
        
        # Check for order block violations
        logger.debug("trade_update: checking OB violations")
        self.check_ob_violations()
        
        # Check stop loss and take profit levels
        logger.debug("trade_update: checking exit conditions")
        self.check_exit_conditions()
    
    
    # =========================
    # Candle Management
    # =========================
    def update_candles(self):
        """Update candle data when new trade occurs"""
        logger.debug("update_candles: called")
        if not self.candles:
            logger.debug("update_candles: no candles present yet")
            return

        current_time = self.epoch_now
        last_candle = self.candles[-1]

        # Determine if we need a new candle based on timeframe
        candle_duration = self.get_candle_duration_ms()

        # Try the string timeframe first (e.g. '1h') to avoid repeated KeyError spam
        # we've observed when attempting ms/seconds before the string key.
        attempts = []
        if hasattr(self, 'CANDLE_LEVEL') and isinstance(self.CANDLE_LEVEL, str):
            attempts.append(self.CANDLE_LEVEL)
        # fall back to seconds and milliseconds if the string key isn't accepted
        try:
            secs = int(candle_duration // 1000)
            attempts.append(secs)
        except Exception:
            secs = None
        try:
            attempts.append(int(candle_duration))
        except Exception:
            pass

        candle_start = None
        last_exc = None
        for attempt in attempts:
            try:
                logger.debug(f"update_candles: trying candle_bin with key={attempt!r} (type={type(attempt).__name__})")
                candle_start = self.candle_bin(current_time, attempt)
                logger.info(f"update_candles: candle_bin succeeded with key={attempt!r}")
                last_exc = None
                break
            except KeyError as e:
                # Only warn for KeyError and continue to next fallback
                logger.debug(f"update_candles: candle_bin KeyError for key={attempt!r}; trying next fallback")
                last_exc = e
                continue
            except Exception as e:
                # Non-KeyError exceptions may indicate different problems; log and continue
                logger.exception(f"update_candles: candle_bin exception for key={attempt!r}: {e}")
                last_exc = e
                continue

        if candle_start is None:
            logger.error(f"update_candles: all candle_bin fallbacks failed. attempts={[(a, type(a).__name__) for a in attempts]} last_exc={last_exc}")
            return
        
        if candle_start > last_candle.time:
            # Start new candle
            logger.info(f"update_candles: new candle starting at {candle_start}")
            new_candle = Candle(
                time=candle_start,
                open=self.current_price,
                high=self.current_price,
                low=self.current_price,
                close=self.current_price,
                volume=0
            )
            self.candles.append(new_candle)
            
            # Process order blocks on new candle
            self.process_new_candle()
        else:
            # Update current candle
            last_candle.high = max(last_candle.high, self.current_price)
            last_candle.low = min(last_candle.low, self.current_price)
            last_candle.close = self.current_price
            logger.debug(f"update_candles: updated last candle close={last_candle.close} high={last_candle.high} low={last_candle.low}")
    
    
    def get_candle_duration_ms(self) -> int:
        """Convert candle level to milliseconds"""
        level_map = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000
        }
        return level_map.get(self.CANDLE_LEVEL, 3600000)
    
    
    # =========================
    # Order Block Detection
    # =========================
    def detect_order_blocks_batch(self):
        """Process all candles to detect initial order blocks"""
        if len(self.candles) < self.SWING_LENGTH * 2:
            return
        
        candle_list = list(self.candles)
        
        # Compute pivots
        for i in range(self.SWING_LENGTH, len(candle_list) - self.SWING_LENGTH):
            self.check_pivot_high(candle_list, i)
            self.check_pivot_low(candle_list, i)
    
    
    def process_new_candle(self):
        """Process newly completed candle for order blocks"""
        if len(self.candles) < self.SWING_LENGTH * 2 + 1:
            return
        
        candle_list = list(self.candles)
        current_idx = len(candle_list) - 1
        
        # Check for pivot confirmation at current index
        self.check_pivot_high(candle_list, current_idx - self.SWING_LENGTH)
        self.check_pivot_low(candle_list, current_idx - self.SWING_LENGTH)
        
        # Check for breakouts
        self.check_breakouts(candle_list, current_idx)
    
    
    def check_pivot_high(self, candles: List[Candle], center_idx: int):
        """Check if candle at center_idx is a pivot high"""
        if center_idx < self.SWING_LENGTH or center_idx >= len(candles) - self.SWING_LENGTH:
            return
        
        center_high = candles[center_idx].high
        
        # Check left side
        left_max = max(c.high for c in candles[center_idx - self.SWING_LENGTH:center_idx])
        # Check right side
        right_max = max(c.high for c in candles[center_idx + 1:center_idx + 1 + self.SWING_LENGTH])
        
        if center_high > left_max and center_high > right_max:
            self.last_pivot_high = {
                'price': center_high,
                'time': candles[center_idx].time,
                'crossed': False
            }
    
    
    def check_pivot_low(self, candles: List[Candle], center_idx: int):
        """Check if candle at center_idx is a pivot low"""
        if center_idx < self.SWING_LENGTH or center_idx >= len(candles) - self.SWING_LENGTH:
            return
        
        center_low = candles[center_idx].low
        
        # Check left side
        left_min = min(c.low for c in candles[center_idx - self.SWING_LENGTH:center_idx])
        # Check right side
        right_min = min(c.low for c in candles[center_idx + 1:center_idx + 1 + self.SWING_LENGTH])
        
        if center_low < left_min and center_low < right_min:
            self.last_pivot_low = {
                'price': center_low,
                'time': candles[center_idx].time,
                'crossed': False
            }
    
    
    def check_breakouts(self, candles: List[Candle], current_idx: int):
        """Check for breakouts and create order blocks"""
        current_candle = candles[current_idx]
        
        # Check for break below pivot low (bearish OB)
        if self.last_pivot_low and not self.last_pivot_low['crossed']:
            if current_candle.close < self.last_pivot_low['price']:
                self.create_bearish_ob(candles, current_idx)
                self.last_pivot_low['crossed'] = True
        
        # Check for break above pivot high (bullish OB)
        if self.last_pivot_high and not self.last_pivot_high['crossed']:
            if current_candle.close > self.last_pivot_high['price']:
                self.create_bullish_ob(candles, current_idx)
                self.last_pivot_high['crossed'] = True
    
    
    def create_bearish_ob(self, candles: List[Candle], breakout_idx: int):
        """Create bearish order block after downside breakout"""
        # Find highest green candle in previous swing_length bars
        best_candle = None
        best_high = -float('inf')
        
        for i in range(max(0, breakout_idx - self.SWING_LENGTH), breakout_idx):
            c = candles[i]
            if c.close > c.open and c.high > best_high:
                best_high = c.high
                best_candle = c
        
        if not best_candle:
            # Fallback: use highest candle regardless of color
            for i in range(max(0, breakout_idx - self.SWING_LENGTH), breakout_idx):
                if candles[i].high > best_high:
                    best_high = candles[i].high
                    best_candle = candles[i]
        
        if best_candle:
            # Calculate strengths
            bullish_vol, bearish_vol = self.calculate_strengths(
                candles, breakout_idx - 1, self.SWING_LENGTH
            )
            total_vol = bullish_vol + bearish_vol
            
            if total_vol < self.MIN_TOTAL_VOLUME:
                return
            
            # Create OB
            ob = OrderBlock(
                kind='bearish',
                top=best_candle.high,
                btm=best_candle.low,
                start_time=best_candle.time,
                create_time=candles[breakout_idx].time,
                bullish_str=bullish_vol,
                bearish_str=bearish_vol,
                vol=best_candle.volume
            )
            
            # Check overlap
            if self.HIDE_OVERLAP and self.has_overlap(ob):
                return
            
            self.order_blocks.append(ob)
            print(f"✓ Bearish OB created @ {ob.top:.2f}-{ob.btm:.2f}")
            
            # Generate signal if strength condition met
            if bearish_vol / total_vol >= self.MIN_STRENGTH_RATIO:
                self.generate_signal('sell', candles[breakout_idx], ob)
    
    
    def create_bullish_ob(self, candles: List[Candle], breakout_idx: int):
        """Create bullish order block after upside breakout"""
        # Find lowest red candle in previous swing_length bars
        best_candle = None
        best_low = float('inf')
        
        for i in range(max(0, breakout_idx - self.SWING_LENGTH), breakout_idx):
            c = candles[i]
            if c.close < c.open and c.low < best_low:
                best_low = c.low
                best_candle = c
        
        if not best_candle:
            # Fallback: use lowest candle regardless of color
            for i in range(max(0, breakout_idx - self.SWING_LENGTH), breakout_idx):
                if candles[i].low < best_low:
                    best_low = candles[i].low
                    best_candle = candles[i]
        
        if best_candle:
            # Calculate strengths
            bullish_vol, bearish_vol = self.calculate_strengths(
                candles, breakout_idx - 1, self.SWING_LENGTH
            )
            total_vol = bullish_vol + bearish_vol
            
            if total_vol < self.MIN_TOTAL_VOLUME:
                return
            
            # Create OB
            ob = OrderBlock(
                kind='bullish',
                top=best_candle.high,
                btm=best_candle.low,
                start_time=best_candle.time,
                create_time=candles[breakout_idx].time,
                bullish_str=bullish_vol,
                bearish_str=bearish_vol,
                vol=best_candle.volume
            )
            
            # Check overlap
            if self.HIDE_OVERLAP and self.has_overlap(ob):
                return
            
            self.order_blocks.append(ob)
            print(f"✓ Bullish OB created @ {ob.top:.2f}-{ob.btm:.2f}")
            
            # Generate signal if strength condition met
            if bullish_vol / total_vol >= self.MIN_STRENGTH_RATIO:
                self.generate_signal('buy', candles[breakout_idx], ob)
    
    
    def calculate_strengths(self, candles: List[Candle], end_idx: int, lookback: int) -> tuple:
        """Calculate bullish and bearish volume strengths"""
        bullish_vol = 0.0
        bearish_vol = 0.0
        
        start_idx = max(0, end_idx - lookback + 1)
        
        for i in range(start_idx, end_idx + 1):
            c = candles[i]
            if c.open > c.close:
                bearish_vol += c.volume
            else:
                bullish_vol += c.volume
        
        return bullish_vol, bearish_vol
    
    
    def has_overlap(self, new_ob: OrderBlock) -> bool:
        """Check if new OB overlaps with existing active OBs"""
        for ob in self.order_blocks:
            if ob.active and ob.kind == new_ob.kind:
                if not (new_ob.top < ob.btm or new_ob.btm > ob.top):
                    return True
        return False
    
    
    def check_ob_violations(self):
        """Check if any active OBs are violated by current price"""
        logger.debug(f"check_ob_violations: current_price={self.current_price} active_obs={len([o for o in self.order_blocks if o.active])}")
        for ob in self.order_blocks:
            if not ob.active:
                continue
            
            violated = False
            if ob.kind == 'bullish':
                if self.VIOLATION_TYPE == 'Close':
                    violated = self.current_price < ob.btm
                else:  # Wick
                    violated = self.current_bid < ob.btm
            else:  # bearish
                if self.VIOLATION_TYPE == 'Close':
                    violated = self.current_price > ob.top
                else:  # Wick
                    violated = self.current_ask > ob.top
            
            if violated:
                ob.active = False
                ob.violated_time = self.epoch_now
                logger.info(f"check_ob_violations: {ob.kind.capitalize()} OB violated @ price={self.current_price:.2f}")
                print(f"✗ {ob.kind.capitalize()} OB violated @ {self.current_price:.2f}")
    
    
    # =========================
    # Signal Generation & Execution
    # =========================
    def generate_signal(self, signal_type: str, candle: Candle, ob: OrderBlock):
        """Generate trading signal and execute if conditions met"""
        logger.info(f"generate_signal: type={signal_type} candle_time={candle.time} ob={ob.top:.2f}-{ob.btm:.2f}")
        # Check if we can open new position
        if len(self.open_positions) >= self.MAX_CONCURRENT_POSITIONS:
            print(f"⚠ Max positions reached ({self.MAX_CONCURRENT_POSITIONS}), skipping signal")
            return
        
        # Calculate position parameters
        ob_size = ob.top - ob.btm
        sl_distance = ob_size * self.STOP_LOSS_MULTIPLIER
        tp_distance = sl_distance * self.TAKE_PROFIT_MULTIPLIER
        
        if signal_type == 'buy':
            entry_price = candle.close
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # sell
            entry_price = candle.close
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Require known live capital to size positions unless USE_FIXED_CAPITAL is enabled.
        if not self.USE_FIXED_CAPITAL and self.current_capital is None:
            logger.warning('generate_signal: current_capital unknown; skipping signal until account balance is available or enable USE_FIXED_CAPITAL')
            return
        capital_to_use = self.INITIAL_CAPITAL if self.USE_FIXED_CAPITAL else self.current_capital
        risk_amount = capital_to_use * (self.RISK_PER_TRADE_PERCENT / 100)
        position_size = risk_amount / sl_distance
        
        # Limit position size
        position_value = position_size * entry_price
        if position_value > self.MAX_POSITION_SIZE_USD:
            position_size = self.MAX_POSITION_SIZE_USD / entry_price
            risk_amount = position_size * sl_distance
        
        # Create position object
        position = TradingPosition(
            entry_time=self.epoch_now,
            entry_price=entry_price,
            position_type='long' if signal_type == 'buy' else 'short',
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ob_size=ob_size,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            capital_at_risk=risk_amount
        )
        
        # Execute order via ProfitView API
        self.execute_order(position, signal_type)
        
        # Publish signal to websocket
        self.publish('signal', {
            'type': signal_type,
            'price': entry_price,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time': self.iso_now
        })
        
        print(f"📈 {signal_type.upper()} signal @ {entry_price:.2f} | Size: {position_size:.4f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
    
    
    def execute_order(self, position: TradingPosition, side: str):
        """Execute market order through ProfitView API"""
        logger.info(f"execute_order: submitting {side} size={position.position_size}")
        response = self.create_market_order(
            venue=self.VENUE,
            sym=self.SYMBOL,
            side='Buy' if side == 'buy' else 'Sell',
            size=position.position_size
        )
        logger.debug(f"execute_order: response_type={type(response)}")
        if not isinstance(response, dict) or response.get('error'):
            logger.error(f"execute_order: order submission failed: {response}")
            print(f"✗ Order failed: {response}")
            return

        order_data = response.get('data', {})
        order_id = order_data.get('order_id')
        if not order_id:
            logger.error(f"execute_order: missing order_id in response: {order_data}")
            print(f"✗ Order failed: missing order_id")
            return

        # Store in pending orders
        self.pending_orders[order_id] = position
        logger.info(f"execute_order: Order submitted: {order_id}")
        print(f"✓ Order submitted: {order_id}")
    
    
    # =========================
    # Position Management
    # =========================
    def check_exit_conditions(self):
        """Check stop loss and take profit conditions for open positions"""
        for position in list(self.open_positions):
            should_exit = False
            exit_reason = None
            exit_price = None
            
            if position.position_type == 'long':
                # Check SL
                if self.current_bid <= position.stop_loss:
                    should_exit = True
                    exit_reason = 'trailing_stop' if position.trailing_active else 'SL'
                    exit_price = position.stop_loss
                # Check TP
                elif self.current_ask >= position.take_profit:
                    should_exit = True
                    exit_reason = 'TP'
                    exit_price = position.take_profit
                    
            else:  # short
                # Check SL
                if self.current_ask >= position.stop_loss:
                    should_exit = True
                    exit_reason = 'trailing_stop' if position.trailing_active else 'SL'
                    exit_price = position.stop_loss
                # Check TP
                elif self.current_bid <= position.take_profit:
                    should_exit = True
                    exit_reason = 'TP'
                    exit_price = position.take_profit
            
            if should_exit:
                self.close_position(position, exit_price, exit_reason)
    
    
    def check_trailing_stops(self):
        """Update trailing stops for profitable positions"""
        for position in self.open_positions:
            if position.position_type == 'long':
                # Update highest price
                if position.highest_price is None:
                    position.highest_price = position.entry_price
                position.highest_price = max(position.highest_price, self.current_ask)
                
                # Check trailing activation
                unrealized_profit = position.highest_price - position.entry_price
                if not position.trailing_active and unrealized_profit >= (position.sl_distance * self.TRAILING_STOP_ACTIVATION):
                    position.trailing_active = True
                    position.initial_stop_loss = position.stop_loss
                    position.trailing_activation_time = self.epoch_now
                    print(f"✓ Trailing stop activated for LONG @ {position.highest_price:.2f}")
                
                # Update trailing stop (with buffer check)
                if position.trailing_active:
                    time_since_activation = self.epoch_now - position.trailing_activation_time
                    if time_since_activation >= self.TRAILING_STOP_BUFFER_MS:
                        new_sl = position.highest_price * (1 - self.TRAILING_STOP_PERCENT / 100)
                        if new_sl > position.stop_loss:
                            position.stop_loss = new_sl
                            
            else:  # short
                # Update lowest price
                if position.lowest_price is None:
                    position.lowest_price = position.entry_price
                position.lowest_price = min(position.lowest_price, self.current_bid)
                
                # Check trailing activation
                unrealized_profit = position.entry_price - position.lowest_price
                if not position.trailing_active and unrealized_profit >= (position.sl_distance * self.TRAILING_STOP_ACTIVATION):
                    position.trailing_active = True
                    position.initial_stop_loss = position.stop_loss
                    position.trailing_activation_time = self.epoch_now
                    print(f"✓ Trailing stop activated for SHORT @ {position.lowest_price:.2f}")
                
                # Update trailing stop (with buffer check)
                if position.trailing_active:
                    time_since_activation = self.epoch_now - position.trailing_activation_time
                    if time_since_activation >= self.TRAILING_STOP_BUFFER_MS:
                        new_sl = position.lowest_price * (1 + self.TRAILING_STOP_PERCENT / 100)
                        if new_sl < position.stop_loss:
                            position.stop_loss = new_sl
    
    
    def close_position(self, position: TradingPosition, exit_price: float, exit_reason: str):
        """Close position and calculate P&L"""
        # Execute closing order
        side = 'Sell' if position.position_type == 'long' else 'Buy'
        response = self.create_market_order(
            venue=self.VENUE,
            sym=self.SYMBOL,
            side=side,
            size=position.position_size
        )
        
        if response['error']:
            print(f"✗ Close order failed: {response['error']['message']}")
            return
        
        # Calculate P&L
        position.exit_time = self.epoch_now
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        
        if position.position_type == 'long':
            position.pnl = exit_price - position.entry_price
        else:
            position.pnl = position.entry_price - exit_price
        
        position.pnl_percent = (position.pnl / position.entry_price) * 100
        
        # Calculate fees
        entry_fee = position.entry_price * position.position_size * (self.COMMISSION_PERCENT + self.SLIPPAGE_PERCENT) / 100
        exit_fee = exit_price * position.position_size * (self.COMMISSION_PERCENT + self.SLIPPAGE_PERCENT) / 100
        total_fees = entry_fee + exit_fee
        
        position.pnl_dollars = (position.pnl * position.position_size) - total_fees
        
        # Update capital. If current capital isn't known yet, defer applying the PnL rather than seeding with a guessed value.
        if self.current_capital is None:
            logger.debug('close_position: current_capital unknown; deferring PnL application')
            if not hasattr(self, '_deferred_capital_adjustments'):
                self._deferred_capital_adjustments = []
            self._deferred_capital_adjustments.append(position.pnl_dollars)
        else:
            self.current_capital += position.pnl_dollars
        
        # Move to closed positions
        self.open_positions.remove(position)
        self.positions.append(position)
        
        # Publish to websocket
        self.publish('position_closed', {
            'type': position.position_type,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': position.pnl_dollars,
            'pnl_percent': position.pnl_percent,
            'time': self.iso_now
        })
        
        print(f"✓ Position closed: {position.position_type.upper()} | PnL: ${position.pnl_dollars:.2f} ({position.pnl_percent:.2f}%) | Reason: {exit_reason}")
    
    
    # =========================
    # HTTP Webhooks
    # =========================
    @http.route
    def get_status(self, data):
        """Get current bot status"""
        return {
            'capital': self.current_capital,
            'capital_known': self.current_capital is not None,
            'open_positions': len(self.open_positions),
            'total_trades': len(self.positions),
            'active_order_blocks': len([ob for ob in self.order_blocks if ob.active]),
            'symbol': self.SYMBOL,
            'venue': self.VENUE
        }
    
    
    @http.route
    def get_positions(self, data):
        """Get all positions"""
        return {
            'open': [{
                'type': p.position_type,
                'entry_price': p.entry_price,
                'size': p.position_size,
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'trailing_active': p.trailing_active
            } for p in self.open_positions],
            'closed': [{
                'type': p.position_type,
                'entry_price': p.entry_price,
                'exit_price': p.exit_price,
                'pnl': p.pnl_dollars,
                'pnl_percent': p.pnl_percent,
                'exit_reason': p.exit_reason
            } for p in self.positions[-10:]]  # Last 10 closed
        }
    
    
    @http.route
    def post_close_all(self, data):
        """Close all open positions"""
        closed_count = 0
        for position in list(self.open_positions):
            self.close_position(position, self.current_price, 'manual')
            closed_count += 1
        
        return {'closed_positions': closed_count}
    
    
    @http.route
    def post_update_params(self, data):
        """Update strategy parameters"""
        if 'risk_percent' in data:
            self.RISK_PER_TRADE_PERCENT = float(data['risk_percent'])
        if 'max_positions' in data:
            self.MAX_CONCURRENT_POSITIONS = int(data['max_positions'])
        if 'trailing_percent' in data:
            self.TRAILING_STOP_PERCENT = float(data['trailing_percent'])
        
        return {'status': 'parameters updated'}