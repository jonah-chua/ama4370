from profitview import Link, http, logger
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import deque


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
    
    # Trading configuration
    SRC = 'woo'
    VENUE = 'WooPaper'
    SYMBOL = 'PERP_BTC_USDT'
    
    # Strategy parameters
    SWING_LENGTH = 10
    VIOLATION_TYPE = "Close"  # "Close" or "Wick"
    HIDE_OVERLAP = True
    MIN_STRENGTH_RATIO = 0.30
    MIN_TOTAL_VOLUME = 1e-7
    
    # Position management
    STOP_LOSS_MULTIPLIER = 0.5
    TAKE_PROFIT_MULTIPLIER = 2.0
    MAX_CONCURRENT_POSITIONS = 2
    TRAILING_STOP_ACTIVATION = 1.5
    TRAILING_STOP_PERCENT = 3.0
    TRAILING_STOP_BUFFER_MS = 3600000  # 1 hour
    
    # Capital management
    RISK_PER_TRADE_PERCENT = 3.0
    MAX_POSITION_SIZE_USD = 100000.0
    FALLBACK_CAPITAL = 0  # Only used if can't fetch from exchange
    
    # Fees (will be updated from exchange)
    COMMISSION_PERCENT = 0.1
    SLIPPAGE_PERCENT = 0.05
    
    # Candle settings
    CANDLE_LEVEL = "1h"
    
    # System control
    init = True  # for first run cleanup
    running = True  # pause/resume strategy
    
    def __init__(self):
        """Initialize strategy parameters and data structures"""
        # Initialize instance attributes BEFORE super().__init__() 
        # because ProfitView can trigger callbacks immediately!
        
        # State management
        self.current_capital = None  # Will be fetched from exchange
        self.candles: deque = deque(maxlen=200)
        self.order_blocks: List[OrderBlock] = []
        self.positions: List[TradingPosition] = []
        self.open_positions: List[TradingPosition] = []
        
        # Pivot tracking
        self.last_pivot_high = None
        self.last_pivot_low = None
        
        # Current market state
        self.current_bid = None
        self.current_ask = None
        self.current_price = None
        
        # Order tracking
        self.pending_orders: Dict[str, TradingPosition] = {}
        
        # Fee tracking
        self.total_fees_paid = 0.0
        
        # Initialization flags - CRITICAL: set these FIRST
        self.initialized = False
        self._fetched_account_info = False
        self._initializing = False  # Lock to prevent concurrent initialization
        
        # NOW initialize parent Link class
        super().__init__()
        
        logger.info(f"Trading bot initialized - {self.SYMBOL} on {self.VENUE}")
        logger.info(f"Risk per trade: {self.RISK_PER_TRADE_PERCENT}% | Max positions: {self.MAX_CONCURRENT_POSITIONS}")
        logger.info(f"⏳ Waiting for market data... Make sure {self.SYMBOL} is subscribed in ProfitView!")
        logger.info(f"💡 If stuck, check: Settings → Market Data → Subscribe to {self.SYMBOL}")
    
    
    # =========================
    # Account & Initialization
    # =========================
    def fetch_account_info(self):
        """Fetch account balance and fee info from exchange (best-effort)"""
        if getattr(self, '_fetched_account_info', False):
            return
        
        try:
            # Try to fetch balances
            resp = self.fetch_balances(self.VENUE)
            
            if resp and not resp.get('error'):
                data = resp.get('data', [])
                
                # Look for USD/USDT balance
                for balance in data:
                    asset = balance.get('asset', '')
                    if asset in ['USD', 'USDT', 'BUSD', 'USDC']:
                        amount = balance.get('amount', 0)
                        if amount > 0:
                            self.current_capital = float(amount)
                            logger.info(f"✓ Fetched capital: ${self.current_capital:.2f} {asset}")
                            break
                
                # If still no capital, try first balance
                if self.current_capital is None and data:
                    self.current_capital = float(data[0].get('amount', self.FALLBACK_CAPITAL))
                    logger.info(f"✓ Using first balance: ${self.current_capital:.2f}")
            
            # Try to get fee rate
            # Note: Some exchanges include fee info in account/balance response
            if resp and isinstance(resp.get('data'), dict):
                acc = resp['data']
                for key in ['maker_fee', 'taker_fee', 'fee', 'commission']:
                    if key in acc:
                        try:
                            fee = float(acc[key])
                            if 0 < fee < 1:  # Assume it's a percentage
                                self.COMMISSION_PERCENT = fee * 100
                            elif fee >= 1:  # Already in percentage
                                self.COMMISSION_PERCENT = fee
                            logger.info(f"✓ Fetched commission: {self.COMMISSION_PERCENT}%")
                            break
                        except:
                            pass
            
            # Fallback to default capital if fetch failed
            if self.current_capital is None:
                self.current_capital = self.FALLBACK_CAPITAL
                logger.warning(f"⚠ Could not fetch balance, using fallback: ${self.current_capital:.2f}")
            
        except Exception as e:
            logger.exception(f"fetch_account_info error: {e}")
            self.current_capital = self.FALLBACK_CAPITAL
            logger.warning(f"⚠ Using fallback capital: ${self.current_capital:.2f}")
        
        finally:
            self._fetched_account_info = True
    
    
    def has_positions(self) -> bool:
        """Check if account has any open positions"""
        try:
            positions = self.fetch_positions(self.VENUE)
            if positions and not positions.get('error'):
                return len(positions.get('data', [])) > 0
        except Exception as e:
            logger.error(f"has_positions error: {e}")
        return False
    
    
    def close_all_positions(self):
        """Close all open positions (used on startup if init=True)"""
        try:
            positions = self.fetch_positions(self.VENUE)
            if not positions or positions.get('error'):
                return
            
            for position in positions['data']:
                # Close by sending opposite side order
                side = 'Buy' if position['side'] == 'Sell' else 'Sell'
                size = abs(float(position.get('pos_size', 0)))
                
                if size > 0:
                    logger.info(f"Closing position: {side} {position['sym']} size={size}")
                    self.create_market_order(
                        venue=self.VENUE,
                        sym=position['sym'],
                        side=side,
                        size=size
                    )
        except Exception as e:
            logger.exception(f"close_all_positions error: {e}")
    
    
    def initialize_candles(self):
        """Fetch historical candles to populate initial state"""
        # Check if already initialized or currently initializing
        if getattr(self, 'initialized', False):
            return
        
        # Prevent concurrent initialization (lock)
        if getattr(self, '_initializing', False):
            logger.debug("initialize_candles: already initializing, skipping")
            return
        
        self._initializing = True
        
        try:
            logger.info(f"Fetching historical candles for {self.SYMBOL}...")
            
            response = self.fetch_candles(
                venue=self.VENUE,
                sym=self.SYMBOL,
                level=self.CANDLE_LEVEL,
                since=self.epoch_now - (200 * 3600000)  # Last 200 hours for 1h
            )
            
            if response.get('error'):
                logger.error(f"Error fetching candles: {response['error']}")
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
            logger.info("✓ Bot ready to trade!")
            
        except Exception as e:
            logger.exception(f"initialize_candles error: {e}")
        finally:
            self._initializing = False
    
    
    # =========================
    # Event Callbacks
    # =========================
    def order_update(self, src: str, sym: str, data: dict):
        """Handle order status updates"""
        order_id = data.get('order_id')
        remain_size = data.get('remain_size', 0)
        
        # Check if order is filled
        if remain_size == 0 and order_id in self.pending_orders:
            position = self.pending_orders[order_id]
            position.order_id = order_id
            self.open_positions.append(position)
            del self.pending_orders[order_id]
            
            logger.info(f"✓ Position opened: {position.position_type.upper()} {position.position_size} @ {position.entry_price}")
    
    
    def fill_update(self, src: str, sym: str, data: dict):
        """Handle trade fill updates and track fees"""
        fee = data.get('fee') or data.get('fee_amount') or data.get('fee_value')
        
        if fee:
            try:
                fee_amt = float(fee)
                self.total_fees_paid += fee_amt
                
                # Deduct fee from capital if known
                if self.current_capital is not None:
                    self.current_capital -= fee_amt
                    logger.info(f"Fee recorded: ${fee_amt:.2f} | Total fees: ${self.total_fees_paid:.2f} | Capital: ${self.current_capital:.2f}")
            except Exception as e:
                logger.exception(f"fill_update fee tracking error: {e}")
        
        logger.info(f"Fill: {data.get('side')} {data.get('fill_size')} @ {data.get('fill_price')}")
    
    
    def position_update(self, src: str, sym: str, data: dict):
        """Handle position updates from exchange"""
        # Can use this for reconciliation if needed
        pass
    
    
    def quote_update(self, src: str, sym: str, data: dict):
        """Handle top-of-book quote updates"""
        if sym != self.SYMBOL:
            return
        
        self.current_bid, _ = data['bid']
        self.current_ask, _ = data['ask']
        self.current_price = (self.current_bid + self.current_ask) / 2
        
        # Log quote updates occasionally (every 10th update to avoid spam)
        if not hasattr(self, '_quote_count'):
            self._quote_count = 0
        self._quote_count += 1
        if self._quote_count % 10 == 0:
            logger.info(f"Quote: bid={self.current_bid:.2f} ask={self.current_ask:.2f} mid={self.current_price:.2f}")
        
        # Check trailing stops if initialized (use getattr to avoid AttributeError)
        if getattr(self, 'initialized', False) and getattr(self, 'running', True):
            self.check_trailing_stops()
    
    
    def trade_update(self, src: str, sym: str, data: dict):
        """Handle market trade updates - main strategy logic"""
        if sym != self.SYMBOL:
            return
        
        # Log trade updates
        logger.info(f"Trade: {data.get('side')} {data.get('size')} @ {data.get('price')}")
        
        # One-time: close existing positions on first run (use getattr to avoid AttributeError)
        if getattr(self, 'init', True) and self.has_positions():
            self.close_all_positions()
        self.init = False
        
        # One-time: fetch account info (use getattr to avoid AttributeError)
        if not getattr(self, '_fetched_account_info', False):
            self.fetch_account_info()
        
        # Initialize candles on first trade (use getattr to avoid AttributeError)
        if not getattr(self, 'initialized', False):
            self.initialize_candles()
            if not getattr(self, 'initialized', False):
                return
        
        # Main trading logic (only if running)
        if getattr(self, 'running', True):
            # Update current price
            self.current_price = data['price']
            
            # Update candles
            self.update_candles()
            
            # Check for order block violations
            self.check_ob_violations()
            
            # Check stop loss and take profit levels
            self.check_exit_conditions()
    
    
    # =========================
    # Candle Management
    # =========================
    def update_candles(self):
        """Update candle data when new trade occurs"""
        if not self.candles:
            return
        
        current_time = self.epoch_now
        last_candle = self.candles[-1]
        
        # Determine if we need a new candle - use string format '1h' instead of milliseconds
        candle_start = self.candle_bin(current_time, self.CANDLE_LEVEL)
        
        if candle_start > last_candle.time:
            # Start new candle
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
        
        # Check for pivot confirmation
        self.check_pivot_high(candle_list, current_idx - self.SWING_LENGTH)
        self.check_pivot_low(candle_list, current_idx - self.SWING_LENGTH)
        
        # Check for breakouts
        self.check_breakouts(candle_list, current_idx)
    
    
    def check_pivot_high(self, candles: List[Candle], center_idx: int):
        """Check if candle at center_idx is a pivot high"""
        if center_idx < self.SWING_LENGTH or center_idx >= len(candles) - self.SWING_LENGTH:
            return
        
        center_high = candles[center_idx].high
        
        # Check left and right sides
        left_max = max(c.high for c in candles[center_idx - self.SWING_LENGTH:center_idx])
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
        
        # Check left and right sides
        left_min = min(c.low for c in candles[center_idx - self.SWING_LENGTH:center_idx])
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
            logger.info(f"✓ Bearish OB created @ {ob.top:.2f}-{ob.btm:.2f}")
            
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
            logger.info(f"✓ Bullish OB created @ {ob.top:.2f}-{ob.btm:.2f}")
            
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
                logger.info(f"✗ {ob.kind.capitalize()} OB violated @ {self.current_price:.2f}")
    
    
    # =========================
    # Signal Generation & Execution
    # =========================
    def generate_signal(self, signal_type: str, candle: Candle, ob: OrderBlock):
        """Generate trading signal and execute if conditions met"""
        # Check if we can open new position
        if len(self.open_positions) >= self.MAX_CONCURRENT_POSITIONS:
            logger.warning(f"Max positions reached ({self.MAX_CONCURRENT_POSITIONS}), skipping signal")
            return
        
        # Require capital info
        if self.current_capital is None:
            logger.warning("Current capital unknown, skipping signal")
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
        
        # Calculate position size based on risk
        risk_amount = self.current_capital * (self.RISK_PER_TRADE_PERCENT / 100)
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
        
        # Execute order
        self.execute_order(position, signal_type)
        
        logger.info(f"📈 {signal_type.upper()} signal @ {entry_price:.2f} | Size: {position_size:.4f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
    
    
    def execute_order(self, position: TradingPosition, side: str):
        """Execute market order through ProfitView API"""
        response = self.create_market_order(
            venue=self.VENUE,
            sym=self.SYMBOL,
            side='Buy' if side == 'buy' else 'Sell',
            size=position.position_size
        )
        
        if response.get('error'):
            logger.error(f"✗ Order failed: {response['error']}")
            return
        
        order_data = response.get('data', {})
        order_id = order_data.get('order_id')
        
        if order_id:
            # Store in pending orders
            self.pending_orders[order_id] = position
            logger.info(f"✓ Order submitted: {order_id}")
    
    
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
                    logger.info(f"✓ Trailing stop activated for LONG @ {position.highest_price:.2f}")
                
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
                    logger.info(f"✓ Trailing stop activated for SHORT @ {position.lowest_price:.2f}")
                
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
        
        if response.get('error'):
            logger.error(f"✗ Close order failed: {response['error']}")
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
        
        # Update capital
        if self.current_capital is not None:
            self.current_capital += position.pnl_dollars
        
        # Move to closed positions
        self.open_positions.remove(position)
        self.positions.append(position)
        
        logger.info(f"✓ Position closed: {position.position_type.upper()} | PnL: ${position.pnl_dollars:.2f} ({position.pnl_percent:.2f}%) | Reason: {exit_reason}")
    
    
    # =========================
    # HTTP Webhooks
    # =========================
    @http.route
    def get_status(self, data):
        """Get current bot status"""
        return {
            'capital': self.current_capital,
            'open_positions': len(self.open_positions),
            'total_trades': len(self.positions),
            'active_order_blocks': len([ob for ob in self.order_blocks if ob.active]),
            'total_fees_paid': self.total_fees_paid,
            'symbol': self.SYMBOL,
            'venue': self.VENUE,
            'running': self.running
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
    def get_pause(self, data):
        """Pause the strategy"""
        self.running = False
        logger.info("Strategy paused")
        return "Paused"
    
    
    @http.route
    def get_resume(self, data):
        """Resume the strategy"""
        self.running = True
        logger.info("Strategy resumed")
        return "Resumed"
