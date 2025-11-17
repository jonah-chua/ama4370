from profitview import Link, http, logger
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import deque
import threading
import numpy as np
import time


# =========================
# Data Structures
# =========================
@dataclass
class Candle:
    time: int  # epoch timestamp in milliseconds
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
    trailing_activation_idx: Optional[int] = None  # Candle index when trailing was activated
    last_trailing_update_price: Optional[float] = None  # Last price where trailing was updated
    
    holding_counter: int = 0  # Number of candles held without reinforcement
    
    order_id: Optional[str] = None  # ProfitView order ID
    
    # Actual fill tracking
    actual_entry_price: Optional[float] = None  # Weighted average fill price from exchange
    actual_exit_price: Optional[float] = None  # Weighted average exit fill price
    entry_fees_paid: float = 0.0  # Actual fees from entry fills
    exit_fees_paid: float = 0.0  # Actual fees from exit fills
    is_closing: bool = False  # Flag to track if position is being closed
    
    # Partial fill tracking for weighted averages
    entry_filled_size: float = 0.0  # Cumulative filled size for entry
    exit_filled_size: float = 0.0  # Cumulative filled size for exit


# =========================
# Main Trading Class
# =========================
class Trading(Link):
    """
    ProfitView Order Block Trading Bot
    
    Implements volumetric order block detection with symmetric pivot analysis,
    strength-based signal generation, and comprehensive position management
    including trailing stops and holding period limits.
    
    Based on the logic from new2_testing.py adapted for ProfitView's event-driven architecture.
    """
    
    # ==================
    # Trading Configuration
    # ==================
    SRC = 'woo'
    VENUE = 'WooPaper'
    SYMBOL = 'PERP_ETH_USDT'
    
    # ==================
    # Order Block Detection Parameters
    # ==================
    SWING_LENGTH = 30  # Number of candles for pivot detection (left=right)
    VIOLATION_TYPE = "Close"  # "Close" or "Wick" - how OBs are violated
    MIN_STRENGTH_RATIO = 0.3  # Minimum dominant volume ratio to generate signal
    MIN_TOTAL_VOLUME = 1e-7  # Minimum total volume to avoid division by zero
    
    # Breakout detection
    BREAK_ATR_MULT = 0.02  # Require breakout > this * ATR to confirm
    OB_SEARCH_WINDOW_MULT = 0.3  # Search window = swing_length * this value
    
    # OB SL floor constraints
    ATR_PERIOD_MULT = 1.2  # ATR period = swing_length * this value
    OB_MIN_SL_ATR_MULT = 2.0  # Minimum SL >= this * ATR
    OB_MIN_SL_PCT = 0.01  # Minimum SL >= this % of entry price
    
    # ATR calculation method
    ATR_METHOD = 'highlow_sma'  # 'highlow_sma', 'maxmin_range', or 'close_range'
    RANGE_PERCENT = 1.0  # For range-based ATR methods
    
    # Sideways market handling
    ATR_SIDEWAYS_WINDOW_MULT = 0.3  # Sideways detection window
    ATR_SIDEWAYS_THRESHOLD_PCT = 0.002  # If ATR/price < this, market is sideways
    SIDEWAYS_ACTION = 'skip'  # 'skip' or 'leverage'
    SIDEWAYS_LEVERAGE_MULT = 0.5  # Capital multiplier for sideways markets
    
    # ==================
    # Position Management Parameters
    # ==================
    STOP_LOSS_MULTIPLIER = 1.2  # SL distance = OB_size * this (placed inside OB)
    TAKE_PROFIT_MULTIPLIER = 15.0  # TP distance = SL_distance * this
    MAX_CONCURRENT_POSITIONS = 3  # Maximum number of open positions
    
    # Trailing stop configuration
    TRAILING_STOP_ACTIVATION = 2.0  # Activate trailing after profit = this * SL distance
    TRAILING_STOP_PERCENT = 1.0  # Trail by this % from peak
    TRAILING_STOP_BUFFER_CANDLES = 10  # Wait this many candles after activation
    TRAILING_STOP_UPDATE_THRESHOLD = 0.4  # Update trail if price moves this %
    TRAILING_ATR_MULT = 1.25  # ATR multiplier for trailing stop floor
    
    # Holding period management
    HOLDING_PERIOD_BARS = 480  # Close position after this many candles (0 = disabled)
    
    # ==================
    # Capital Management Parameters
    # ==================
    RISK_PER_TRADE_PERCENT = 1.5  # Risk this % of capital per trade
    USE_FIXED_CAPITAL = True  # If True, don't compound; always use initial capital
    MAX_POSITION_SIZE_USD = 100000.0  # Maximum position value
    FALLBACK_CAPITAL = 5000.0  # Used if fetch_balances fails
    
    # Position sizing floors (per new2_testing to prevent huge positions when SL is tiny)
    MIN_RISK_PCT = 0.002  # Ensure sl_distance is at least this % of entry price (0.2%)
    ATR_HALF_FLOOR = 0.5  # Ensure sl_distance is at least this * ATR / 2
    
    # Max Loss Limit - Trading pauses if reached
    MAX_LOSS_LIMIT = 5000.0  # Maximum loss allowed (positive number)
    # Note: Actual limit is MAX_LOSS_LIMIT minus any cumulative profits earned
    # Example: If you've made $1000 profit, you can lose up to $6000 before pausing
    
    # Entry and fees (will be updated from exchange)
    ENTRY_PRICE_MODE = "Close"  # "close" or "worst" (high for longs, low for shorts)
    COMMISSION_PERCENT = 0.1  # Will be updated from account info
    SLIPPAGE_PERCENT = 0.05  # Slippage estimate for entry price (PnL impact, not a fee)
    
    # EMA-based entry filter
    EMA_PERIOD_MULT = 0.4  # EMA period = swing_length * this value
    ENTRY_DIFF_LONG_PCT = 0.02  # Skip long if price > EMA by this %
    ENTRY_DIFF_SHORT_PCT = 0.03  # Skip short if price < EMA by this %
    
    # ==================
    # Candle Settings
    # ==================
    CANDLE_LEVEL = "15m"  # Candle timeframe
    CANDLE_LEVEL_MS = 15 * 60 * 1000  # 15 minutes in milliseconds
    
    # ==================
    # System Control
    # ==================
    init = True  # Close existing positions on first run
    running = True  # Pause/resume strategy
    
    def __init__(self):
        """Initialize strategy parameters and data structures"""
        
        # CRITICAL: Set ALL attributes BEFORE super().__init__() because
        # ProfitView can trigger callbacks immediately!
        
        # ==================
        # Initialization Flags (MUST BE FIRST!)
        # ==================
        self.initialized = False
        self._fetched_account_info = False
        self._init_lock = threading.Lock()
        self._account_fetch_attempts = 0
        self._max_account_fetch_attempts = 3
        
        # ==================
        # State Management
        # ==================
        self.current_capital = None
        self.initial_capital = None
        self.candles: deque = deque(maxlen=500)
        self.order_blocks: List[OrderBlock] = []
        self.positions: List[TradingPosition] = []
        self.open_positions: List[TradingPosition] = []
        
        # ==================
        # Pivot Tracking
        # ==================
        self.last_pivot_high = None
        self.last_pivot_low = None
        self.pivot_highs: List[tuple] = []
        self.pivot_lows: List[tuple] = []
        # Track pivot indices that have already been used to create an OB
        # This prevents repeated OB creation from the same pivot
        self._used_pivots = set()
        
        # ==================
        # Market State
        # ==================
        self.current_bid = None
        self.current_ask = None
        self.current_price = None
        
        # ==================
        # Order Tracking
        # ==================
        self.pending_orders: Dict[str, TradingPosition] = {}
        
        # ==================
        # Fee Tracking
        # ==================
        self.total_fees_paid = 0.0
        self.fee_baseline = 0.0  # Baseline from fetch_positions at startup
        self.last_known_fees = 0.0  # Most recent fee snapshot from fetch_positions
        
        # ==================
        # PnL & Loss Limit Tracking
        # ==================
        self.cumulative_pnl = 0.0  # Running total of realized PnL (positive = profit, negative = loss)
        self.max_loss_reached = False  # Flag to pause trading when loss limit reached
        
        # ==================
        # Computed Parameters
        # ==================
        self.ob_search_window = max(1, int(self.SWING_LENGTH * self.OB_SEARCH_WINDOW_MULT))
        self.atr_period = max(1, int(self.SWING_LENGTH * self.ATR_PERIOD_MULT))
        self.atr_sideways_window = max(1, int(self.SWING_LENGTH * self.ATR_SIDEWAYS_WINDOW_MULT))
        self.ema_period = max(1, int(self.SWING_LENGTH * self.EMA_PERIOD_MULT))
        
        # ==================
        # Statistics
        # ==================
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0
        }
        
        # NOW initialize parent Link class (this triggers callbacks!)
        super().__init__()
        
        logger.info(f"=== Order Block Trading Bot Initialized ===")
        logger.info(f"Symbol: {self.SYMBOL} on {self.VENUE}")
        logger.info(f"Swing Length: {self.SWING_LENGTH} | Min Strength: {self.MIN_STRENGTH_RATIO}")
        logger.info(f"Risk per Trade: {self.RISK_PER_TRADE_PERCENT}% | Max Positions: {self.MAX_CONCURRENT_POSITIONS}")
        logger.info(f"Candle Level: {self.CANDLE_LEVEL} | Holding Period: {self.HOLDING_PERIOD_BARS} bars")
        logger.info(f"⏳ Waiting for market data... Ensure {self.SYMBOL} is subscribed in ProfitView!")
        logger.info(f"💡 Note: Bot needs LIVE trade updates to function. Historical data loaded successfully.")
        
        # Initialize fee baseline (best effort)
        try:
            self._initialize_fee_baseline()
        except Exception as e:
            logger.warning(f"Failed to initialize fee baseline: {e}")
    
    
    # =========================
    # Account & Initialization
    # =========================
    def _initialize_fee_baseline(self):
        """Fetch initial fee baseline from fetch_positions for delta tracking."""
        try:
            resp = self.fetch_positions(self.VENUE)
            if resp and not resp.get('error'):
                positions = resp.get('data', [])
                for pos in positions:
                    if pos.get('sym') == self.SYMBOL:
                        fees = float(pos.get('fees', 0.0))
                        self.fee_baseline = fees
                        self.last_known_fees = fees
                        logger.info(f"Fee baseline initialized: {fees:.4f} USDT")
                        return
                # No position for this symbol yet
                self.fee_baseline = 0.0
                self.last_known_fees = 0.0
                logger.info("Fee baseline: 0.0 (no existing position)")
        except Exception as e:
            logger.warning(f"Failed to initialize fee baseline: {e}")
            self.fee_baseline = 0.0
            self.last_known_fees = 0.0
    
    def fetch_account_info(self):
        """
        Fetch account balance and fee info from exchange.
        Uses retry logic and proper error handling.
        Per ProfitView docs: fetch_balances returns {'data': [{'asset': str, 'amount': float}]}
        """
        if self._fetched_account_info:
            return
        
        self._account_fetch_attempts += 1
        
        try:
            logger.info(f"Fetching account info (attempt {self._account_fetch_attempts}/{self._max_account_fetch_attempts})...")
            
            # Fetch balances per docs format
            balance_resp = self.fetch_balances(self.VENUE)
            
            if balance_resp and not balance_resp.get('error'):
                balances = balance_resp.get('data', [])
                
                # Look for USD/USDT balance
                for bal in balances:
                    asset = bal.get('asset', '').upper()
                    if asset in ['USD', 'USDT', 'USDC', 'BUSD']:
                        amount = float(bal.get('amount', 0))
                        if amount > 0:
                            self.current_capital = amount
                            if self.initial_capital is None:
                                self.initial_capital = amount
                            logger.info(f"✓ Account balance: {amount:.2f} {asset}")
                            break
                
                if self.current_capital is None:
                    logger.warning(f"No USD balance found. Using fallback: {self.FALLBACK_CAPITAL}")
                    self.current_capital = self.FALLBACK_CAPITAL
                    self.initial_capital = self.FALLBACK_CAPITAL
            else:
                error = balance_resp.get('error') if balance_resp else 'No response'
                logger.warning(f"Failed to fetch balances: {error}")
                
                # Retry logic
                if self._account_fetch_attempts < self._max_account_fetch_attempts:
                    logger.info("Will retry on next opportunity...")
                    return  # Don't set _fetched_account_info, allow retry
                else:
                    logger.warning(f"Max attempts reached. Using fallback capital: {self.FALLBACK_CAPITAL}")
                    self.current_capital = self.FALLBACK_CAPITAL
                    self.initial_capital = self.FALLBACK_CAPITAL
            
            # Try to get commission info (best effort)
            # Note: ProfitView doesn't have a standard fee endpoint, so we use default
            logger.info(f"Using commission: {self.COMMISSION_PERCENT}% | Slippage: {self.SLIPPAGE_PERCENT}%")
            
            self._fetched_account_info = True
            
        except Exception as e:
            logger.exception(f"Error fetching account info: {e}")
            
            # On exception, use fallback if max attempts reached
            if self._account_fetch_attempts >= self._max_account_fetch_attempts:
                logger.warning(f"Using fallback capital: {self.FALLBACK_CAPITAL}")
                self.current_capital = self.FALLBACK_CAPITAL
                self.initial_capital = self.FALLBACK_CAPITAL
                self._fetched_account_info = True
    
    
    def has_positions(self) -> bool:
        """Check if account has any open ETH positions via exchange API"""
        try:
            resp = self.fetch_positions(self.VENUE)
            if resp and not resp.get('error'):
                positions = resp.get('data', [])
                # Only count ETH positions
                eth_positions = [p for p in positions if p['sym'] == self.SYMBOL]
                return len(eth_positions) > 0
        except Exception as e:
            logger.exception(f"Error checking positions: {e}")
        return False
    
    
    def close_all_positions(self):
        """Close all ETH positions only (used on startup if init=True) with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Closing existing ETH positions (attempt {attempt}/{max_retries})...")
                resp = self.fetch_positions(self.VENUE)
                
                if resp and not resp.get('error'):
                    positions = resp.get('data', [])
                    
                    # Filter to only ETH positions
                    eth_positions = [p for p in positions if p['sym'] == self.SYMBOL]
                    
                    if not eth_positions:
                        logger.info("No ETH positions to close")
                        return
                    
                    for pos in eth_positions:
                        side = 'Buy' if pos['side'] == 'Sell' else 'Sell'
                        size = pos['pos_size']
                        
                        logger.info(f"Closing {pos['side']} ETH position: {side} {size}")
                        
                        # Use market order to close with retry
                        close_resp = self.create_market_order(self.VENUE, self.SYMBOL, side=side, size=size)
                        
                        # Check if market unavailable error
                        if close_resp and close_resp.get('error'):
                            error_msg = close_resp['error'].get('message', '')
                            if 'unavailable' in error_msg.lower() and attempt < max_retries:
                                logger.warning(f"Market unavailable, retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                    
                    logger.info(f"Closed {len(eth_positions)} ETH position(s)")
                    return  # Success
                
                elif resp and resp.get('error'):
                    error_msg = resp['error'].get('message', '')
                    if 'unavailable' in error_msg.lower() and attempt < max_retries:
                        logger.warning(f"Market unavailable on fetch_positions, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        logger.error(f"Error fetching positions: {resp.get('error')}")
                        return
            
            except Exception as e:
                logger.exception(f"Error closing positions (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.warning(f"Failed to close positions after {max_retries} attempts")
    
    
    def initialize_candles(self):
        """
        Fetch historical candles to populate initial state.
        Per ProfitView docs: fetch_candles returns {'data': [{'time': int, 'open': float, ...}]}
        """
        # Thread-safe check
        with self._init_lock:
            if self.initialized:
                return
            
            try:
                logger.info("Initializing historical candles...")
                
                # Calculate how far back to fetch (need enough for swing detection + ATR)
                required_candles = self.SWING_LENGTH * 3 + self.atr_period
                logger.info(f"Fetching {required_candles} historical candles (swing={self.SWING_LENGTH}, atr_period={self.atr_period})")
                
                # Fetch historical data - 'since' is in milliseconds
                lookback_ms = required_candles * self.CANDLE_LEVEL_MS
                since_ms = self.epoch_now - lookback_ms
                
                # Log the date range for visibility
                from datetime import datetime, timezone
                since_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                now_dt = datetime.fromtimestamp(self.epoch_now / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                logger.info(f"Fetching candles from {since_dt} to {now_dt} ({lookback_ms/1000/3600:.1f} hours)")
                
                resp = self.fetch_candles(
                    self.VENUE,
                    sym=self.SYMBOL,
                    level=self.CANDLE_LEVEL,  # Use string format like "15m"
                    since=since_ms
                )
                
                if resp and not resp.get('error'):
                    candle_data = resp.get('data', [])
                    
                    if candle_data:
                        # Parse candles per docs format
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
                        
                        # Log detailed info about what was fetched
                        first_candle = self.candles[0]
                        last_candle = self.candles[-1]
                        first_dt = datetime.fromtimestamp(first_candle.time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        last_dt = datetime.fromtimestamp(last_candle.time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        
                        logger.info(f"✓ Loaded {len(self.candles)} historical candles")
                        logger.info(f"  First candle: {first_dt} | O:{first_candle.open:.2f} H:{first_candle.high:.2f} L:{first_candle.low:.2f} C:{first_candle.close:.2f}")
                        logger.info(f"  Last candle:  {last_dt} | O:{last_candle.open:.2f} H:{last_candle.high:.2f} L:{last_candle.low:.2f} C:{last_candle.close:.2f}")
                        
                        # Run initial OB detection
                        self.detect_order_blocks_batch()
                        logger.info(f"✓ Detected {len(self.order_blocks)} initial order blocks")
                        
                        # Log OB details if any were found
                        if self.order_blocks:
                            for i, ob in enumerate(self.order_blocks[:3]):  # Show first 3
                                ob_dt = datetime.fromtimestamp(ob.start_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                                logger.info(f"  OB {i+1}: {ob.kind} @ {ob_dt} | top={ob.top:.2f} btm={ob.btm:.2f} | strength={ob.bullish_str if ob.kind == 'bullish' else ob.bearish_str:.0f}/{ob.vol:.0f}")
                        else:
                            logger.warning("⚠️ No order blocks detected in historical data!")
                            logger.warning(f"  This may indicate: 1) Not enough pivots confirmed yet (need {self.SWING_LENGTH*2} candles), 2) No breakouts occurred, or 3) All OBs filtered out")
                        
                        self.initialized = True
                    else:
                        logger.warning("No historical candles returned")
                else:
                    error = resp.get('error') if resp else 'No response'
                    logger.error(f"Failed to fetch candles: {error}")
                
            except Exception as e:
                logger.exception(f"Error initializing candles: {e}")
    
    
    # =========================
    # Event Callbacks (ProfitView API)
    # =========================
    def order_update(self, src: str, sym: str, data: dict):
        """
        Handle order status updates per ProfitView docs.
        Data format: {'order_id': str, 'remain_size': float, 'side': str, ...}
        """
        if sym != self.SYMBOL:
            return
        
        order_id = data.get('order_id')
        remain_size = data.get('remain_size', 0)
        side = data.get('side')
        
        logger.info(f"Order Update: {side} {data.get('order_size')} @ {data.get('order_price')} | Remain: {remain_size}")
        
        # Try to extract actual fill price from order_update
        if order_id and order_id in self.pending_orders:
            position = self.pending_orders[order_id]
            # Populate actual_entry_price if available and not set
            if position.actual_entry_price is None:
                fill_px = data.get('fill_price') or data.get('order_price')
                if fill_px:
                    try:
                        position.actual_entry_price = float(fill_px)
                        position.entry_filled_size = position.position_size
                        logger.info(f"Set actual_entry_price from order_update: {position.actual_entry_price:.2f}")
                    except (ValueError, TypeError):
                        pass
        
        # Check if order is fully filled
        if remain_size == 0 and order_id in self.pending_orders:
            position = self.pending_orders[order_id]
            logger.info(f"✓ Order filled: {position.position_type} position opened")
            self.open_positions.append(position)
            del self.pending_orders[order_id]
            # Update fee delta after position opened
            self._update_fee_delta()
    
    
    def fill_update(self, src: str, sym: str, data: dict):
        """
        Handle trade fill updates and track fees per ProfitView docs.
        Data format: {'fill_price': float, 'fill_size': float, 'fee': float, 'side': str, 'order_id': str, ...}
        
        This tracks actual fill prices and fees for accurate PnL calculation.
        Handles partial fills with weighted averaging.
        """
        if sym != self.SYMBOL:
            return
        
        order_id = data.get('order_id')
        side = data.get('side')
        
        # Extract fill price - try multiple field names
        fill_price = (data.get('fill_price') or 
                     data.get('fillPrice') or 
                     data.get('price') or 
                     data.get('last_price'))
        
        # Extract fill size - try multiple field names
        fill_size = (data.get('fill_size') or 
                    data.get('fillSize') or 
                    data.get('size') or 
                    data.get('qty') or 
                    data.get('quantity'))
        
        # Extract fee - try multiple field names
        fee = (data.get('fee') or 
               data.get('fee_amount') or 
               data.get('fee_value') or 
               data.get('feeAmount') or 
               data.get('commission'))
        
        fee_amount = 0.0
        if fee:
            try:
                fee_amount = abs(float(fee))  # Ensure positive
            except (ValueError, TypeError):
                logger.warning(f"Could not parse fee: {fee}")
        
        # Update global fee tracking
        if fee_amount > 0:
            self.total_fees_paid += fee_amount
            self.stats['total_fees'] += fee_amount
        
        # Parse fill price and size
        try:
            fill_price_float = float(fill_price) if fill_price else None
            fill_size_float = float(fill_size) if fill_size else None
        except (ValueError, TypeError):
            logger.warning(f"Could not parse fill_price={fill_price} or fill_size={fill_size}")
            fill_price_float = None
            fill_size_float = None
        
        # Track actual fill prices for positions (weighted average for partial fills)
        # Entry fills: position is in pending_orders
        if order_id and order_id in self.pending_orders:
            position = self.pending_orders[order_id]
            
            if fill_price_float and fill_size_float:
                # Weighted average calculation
                if position.actual_entry_price is None:
                    position.actual_entry_price = fill_price_float
                    position.entry_filled_size = fill_size_float
                else:
                    # Weighted average: (old_price * old_size + new_price * new_size) / (old_size + new_size)
                    total_size = position.entry_filled_size + fill_size_float
                    position.actual_entry_price = (
                        (position.actual_entry_price * position.entry_filled_size + 
                         fill_price_float * fill_size_float) / total_size
                    )
                    position.entry_filled_size = total_size
                
                logger.info(f"Entry Fill: {side} {fill_size_float:.4f} @ {fill_price_float:.2f} | "
                          f"Avg Entry: {position.actual_entry_price:.2f}")
            else:
                logger.warning(f"Entry Fill missing price/size data")
        
        # Exit fills: check if any open position is currently closing
        elif fill_price_float and fill_size_float:
            for position in self.open_positions:
                if position.is_closing:
                    # Weighted average calculation for exit
                    if position.actual_exit_price is None:
                        position.actual_exit_price = fill_price_float
                        position.exit_filled_size = fill_size_float
                    else:
                        # Weighted average
                        total_size = position.exit_filled_size + fill_size_float
                        position.actual_exit_price = (
                            (position.actual_exit_price * position.exit_filled_size + 
                             fill_price_float * fill_size_float) / total_size
                        )
                        position.exit_filled_size = total_size
                    
                    logger.info(f"Exit Fill: {side} {fill_size_float:.4f} @ {fill_price_float:.2f} | "
                              f"Avg Exit: {position.actual_exit_price:.2f}")
                    break
        
        # Fallback logging if we couldn't match to a position
        if not order_id or order_id not in self.pending_orders:
            if fill_price_float and fill_size_float:
                logger.info(f"Fill (unmatched): {side} {fill_size_float:.4f} @ {fill_price_float:.2f}")
            else:
                logger.info(f"Fill (unmatched): {side}")
    
    
    def position_update(self, src: str, sym: str, data: dict):
        """
        Handle position updates from exchange per ProfitView docs.
        Data format: {'sym': str, 'side': str, 'pos_size': float, 'entry_price': float, ...}
        """
        if sym != self.SYMBOL:
            return
        
        # This can be used for reconciliation or external position changes
        pos_size = data.get('pos_size', 0)
        side = data.get('side')
        
        if pos_size == 0:
            logger.info(f"Position closed externally: {side}")
            # Could clean up our internal tracking here
        else:
            logger.info(f"Position Update: {side} {pos_size} @ {data.get('entry_price')}")
    
    
    def quote_update(self, src: str, sym: str, data: dict):
        """
        Handle top-of-book quote updates per ProfitView docs.
        Data format: {'bid': [price, size], 'ask': [price, size], 'time': int}
        """
        if sym != self.SYMBOL:
            return
        
        self.current_bid, _ = data['bid']
        self.current_ask, _ = data['ask']
        self.current_price = (self.current_bid + self.current_ask) / 2
        
        # Check trailing stops on quote updates (more frequent than trades)
        if self.initialized and self.running:
            self.check_trailing_stops()
    
    
    def trade_update(self, src: str, sym: str, data: dict):
        """
        Handle market trade updates - main strategy entry point per ProfitView docs.
        Data format: {'side': str, 'price': float, 'size': float, 'time': int}
        """
        if sym != self.SYMBOL:
            return
        
        # Log first trade update
        if not hasattr(self, '_first_trade_logged'):
            logger.info(f"✓ First trade update received for {sym} - bot is now active!")
            self._first_trade_logged = True
        
        # One-time: close existing positions on first run
        if self.init:
            if self.has_positions():
                self.close_all_positions()
            self.init = False
        
        # One-time: fetch account info
        if not self._fetched_account_info:
            self.fetch_account_info()
        
        # Initialize candles on first trade
        if not self.initialized:
            self.initialize_candles()
            if not self.initialized:
                return  # Still not ready
        
        # Reconcile pending orders (check if any were filled without callback)
        if self.pending_orders:
            self._reconcile_pending_orders()
        
        # Main trading logic (only if running)
        if self.running:
            # Extract trade size for volume accumulation
            trade_size = data.get('size', 0.0)
            self.update_candles(trade_size)
            self.check_exit_conditions()
    
    
    def _update_fee_delta(self):
        """Update fee delta from fetch_positions and attribute to recent position."""
        try:
            resp = self.fetch_positions(self.VENUE)
            if resp and not resp.get('error'):
                positions = resp.get('data', [])
                for pos in positions:
                    if pos.get('sym') == self.SYMBOL:
                        current_fees = float(pos.get('fees', 0.0))
                        fee_delta = current_fees - self.last_known_fees
                        
                        if fee_delta > 1e-6:  # Meaningful delta
                            # Attribute to most recent open position (best effort)
                            if self.open_positions:
                                recent_pos = self.open_positions[-1]
                                recent_pos.entry_fees_paid += fee_delta
                                logger.info(f"Fee delta attributed: +{fee_delta:.4f} to position (total: {recent_pos.entry_fees_paid:.4f})")
                            
                            self.total_fees_paid += fee_delta
                            self.stats['total_fees'] += fee_delta
                            self.last_known_fees = current_fees
                        return
        except Exception as e:
            logger.warning(f"Failed to update fee delta: {e}")
    
    def _reconcile_pending_orders(self):
        """Check if pending orders have been filled (reconciliation if callbacks missed)."""
        if not self.pending_orders:
            return
        
        try:
            resp = self.fetch_open_orders(self.VENUE)
            if resp and not resp.get('error'):
                open_order_ids = {o.get('order_id') for o in resp.get('data', [])}
                
                # Check if any pending order is no longer open (was filled)
                filled_orders = []
                for order_id, position in list(self.pending_orders.items()):
                    if order_id not in open_order_ids:
                        filled_orders.append((order_id, position))
                
                for order_id, position in filled_orders:
                    logger.info(f"Reconciliation: order {order_id} was filled (moved to open)")
                    self.open_positions.append(position)
                    del self.pending_orders[order_id]
                    self._update_fee_delta()
        except Exception as e:
            logger.warning(f"Failed to reconcile pending orders: {e}")
    
    # =========================
    # Candle Management
    # =========================
    def update_candles(self, trade_size: float = 0.0):
        """
        Update candle data when new trade occurs.
        Aggregates trades into candles based on CANDLE_LEVEL_MS.
        
        Args:
            trade_size: Volume from the trade_update event to accumulate in the current candle
        """
        if not self.candles:
            return
        
        current_time = self.epoch_now
        last_candle = self.candles[-1]
        
        # Determine if we need a new candle - use string format per ProfitView docs
        candle_start = self.candle_bin(current_time, self.CANDLE_LEVEL, ceil=False)
        
        if candle_start > last_candle.time:
            # New candle period - finalize the last one and process for OBs
            logger.info(f"New candle: {last_candle.time} | O:{last_candle.open:.2f} H:{last_candle.high:.2f} L:{last_candle.low:.2f} C:{last_candle.close:.2f} V:{last_candle.volume:.4f}")
            
            # Process the completed candle for order blocks
            self.process_new_candle()
            
            # Start new candle with current price and initial trade volume
            new_candle = Candle(
                time=candle_start,
                open=self.current_price,
                high=self.current_price,
                low=self.current_price,
                close=self.current_price,
                volume=trade_size
            )
            self.candles.append(new_candle)
        else:
            # Update current candle with price and accumulate volume from this trade
            last_candle.high = max(last_candle.high, self.current_price)
            last_candle.low = min(last_candle.low, self.current_price)
            last_candle.close = self.current_price
            last_candle.volume += trade_size
    
    
    # =========================
    # Order Block Detection (from new2_testing.py)
    # =========================
    def detect_order_blocks_batch(self):
        """Process all candles to detect initial order blocks"""
        if len(self.candles) < self.SWING_LENGTH * 2:
            return
        
        candle_list = list(self.candles)
        
        # Compute pivots for all candles
        for i in range(self.SWING_LENGTH, len(candle_list) - self.SWING_LENGTH):
            self.check_pivot_high(candle_list, i)
            self.check_pivot_low(candle_list, i)
        
        logger.info(f"Batch pivot detection: {len(self.pivot_highs)} highs, {len(self.pivot_lows)} lows")
    
    
    def process_new_candle(self):
        """Process newly completed candle for order blocks"""
        if len(self.candles) < self.SWING_LENGTH * 2 + 1:
            return
        
        # Increment holding_counter for all open positions (Issue #6 fix)
        for position in self.open_positions:
            position.holding_counter += 1
        
        candle_list = list(self.candles)
        current_idx = len(candle_list) - 1
        
        # Check for pivot confirmation at swing_length bars back
        pivot_idx = current_idx
        self.check_pivot_high(candle_list, pivot_idx)
        self.check_pivot_low(candle_list, pivot_idx)
        
        # Check for breakouts at current candle
        self.check_breakouts(candle_list, current_idx)
        
        # Check for OB violations
        self.check_ob_violations()
    
    
    def check_pivot_high(self, candles: List[Candle], center_idx: int):
        """
        Check if candle at center_idx is a pivot high.
        Pivot high: center candle high is highest within swing_length on both sides.
        Per Latest_test.py: pivot is stored at confirmation index (center + SWING_LENGTH).
        """
        if center_idx < self.SWING_LENGTH or center_idx >= len(candles) - self.SWING_LENGTH:
            return
        
        center_high = candles[center_idx].high
        
        # Check if it's the highest in the window
        is_pivot = True
        for i in range(center_idx - self.SWING_LENGTH, center_idx + self.SWING_LENGTH + 1):
            if i != center_idx and candles[i].high >= center_high:
                is_pivot = False
                break
        
        if is_pivot:
            # Store pivot at confirmation index (matches Latest_test.py behavior)
            pivot_idx_confirm = center_idx + self.SWING_LENGTH
            if pivot_idx_confirm < len(candles):
                self.last_pivot_high = (pivot_idx_confirm, center_high, candles[pivot_idx_confirm].time)
                self.pivot_highs.append(self.last_pivot_high)
                logger.info(f"Pivot High detected at center {center_idx}, confirmed at {pivot_idx_confirm}: {center_high:.2f}")
    
    
    def check_pivot_low(self, candles: List[Candle], center_idx: int):
        """
        Check if candle at center_idx is a pivot low.
        Pivot low: center candle low is lowest within swing_length on both sides.
        Per Latest_test.py: pivot is stored at confirmation index (center + SWING_LENGTH).
        """
        if center_idx < self.SWING_LENGTH or center_idx >= len(candles) - self.SWING_LENGTH:
            return
        
        center_low = candles[center_idx].low
        
        # Check if it's the lowest in the window
        is_pivot = True
        for i in range(center_idx - self.SWING_LENGTH, center_idx + self.SWING_LENGTH + 1):
            if i != center_idx and candles[i].low <= center_low:
                is_pivot = False
                break
        
        if is_pivot:
            # Store pivot at confirmation index (matches Latest_test.py behavior)
            pivot_idx_confirm = center_idx + self.SWING_LENGTH
            if pivot_idx_confirm < len(candles):
                self.last_pivot_low = (pivot_idx_confirm, center_low, candles[pivot_idx_confirm].time)
                self.pivot_lows.append(self.last_pivot_low)
                logger.info(f"Pivot Low detected at center {center_idx}, confirmed at {pivot_idx_confirm}: {center_low:.2f}")
    
    
    def check_breakouts(self, candles: List[Candle], current_idx: int):
        """
        Check if current candle breaks a pivot level (BOS/MSB).
        If yes, create appropriate order block.
        """
        if current_idx < self.SWING_LENGTH * 2:
            return
        
        current_candle = candles[current_idx]
        
        # Calculate ATR for breakout threshold
        atr = self.calculate_atr(candles, current_idx)
        breakout_threshold = atr * self.BREAK_ATR_MULT
        
        # Check for bearish breakout (break below pivot low = potential bullish OB)
        if self.last_pivot_low:
            pivot_idx, pivot_price, pivot_time = self.last_pivot_low

            # Skip if this pivot has already been used to create an OB
            if pivot_idx not in self._used_pivots:
                if current_candle.close < (pivot_price - breakout_threshold):
                    logger.info(f"Bearish breakout detected: broke pivot low at {pivot_price:.2f}")
                    self.create_bullish_ob(candles, current_idx, pivot_idx)
        
        # Check for bullish breakout (break above pivot high = potential bearish OB)
        if self.last_pivot_high:
            pivot_idx, pivot_price, pivot_time = self.last_pivot_high

            # Skip if this pivot has already been used to create an OB
            if pivot_idx not in self._used_pivots:
                if current_candle.close > (pivot_price + breakout_threshold):
                    logger.info(f"Bullish breakout detected: broke pivot high at {pivot_price:.2f}")
                    self.create_bearish_ob(candles, current_idx, pivot_idx)
    
    
    def create_bearish_ob(self, candles: List[Candle], breakout_idx: int, pivot_idx: int):
        """
        Create a bearish order block after bullish breakout.
        Per new2_testing: Select highest GREEN/BULLISH candle (close > open) before breakout.
        Choose by highest HIGH (not close).
        """
        # Select candidate candle from the bars immediately BEFORE the breakout
        # Match Latest_test.py semantics: search t-1 .. t-SWING_LENGTH (most recent swing_length bars)
        best_candle_idx = None
        highest_high = -float('inf')
        for j in range(1, self.SWING_LENGTH + 1):
            idx = breakout_idx - j
            if idx < 0:
                break
            c = candles[idx]
            if c.close > c.open and c.high > highest_high:
                highest_high = c.high
                best_candle_idx = idx

        # Fallback: choose the maximum high in the same backward window regardless of color
        if best_candle_idx is None:
            for j in range(1, self.SWING_LENGTH + 1):
                idx = breakout_idx - j
                if idx < 0:
                    break
                c = candles[idx]
                if c.high > highest_high:
                    highest_high = c.high
                    best_candle_idx = idx
        
        if best_candle_idx is None:
            return  # No suitable candle found
        
        ob_candle = candles[best_candle_idx]
        
        # Calculate strengths (use full SWING_LENGTH to match Latest_test behavior)
        bullish_str, bearish_str = self.calculate_strengths(candles, best_candle_idx, self.SWING_LENGTH)
        total_vol = bullish_str + bearish_str
        
        # Use MIN_TOTAL_VOLUME as a floor for ratio calculation, not as a gate
        # Always create OB even if volume is low (per Latest_test.py behavior)
        if total_vol < self.MIN_TOTAL_VOLUME:
            total_vol = self.MIN_TOTAL_VOLUME
        
        # OB geometry: use window around selected candle (matches Latest_test.py)
        half_window = self.ob_search_window // 2
        ws = max(0, best_candle_idx - half_window)
        we = min(len(candles) - 1, best_candle_idx + half_window)
        ob_top = max(c.high for c in candles[ws:we+1])
        ob_btm = min(c.low for c in candles[ws:we+1])
        
        # Create OB (always created, regardless of volume)
        ob = OrderBlock(
            kind="bearish",
            top=ob_top,
            btm=ob_btm,
            start_time=ob_candle.time,
            create_time=candles[breakout_idx].time,
            bullish_str=bullish_str,
            bearish_str=bearish_str,
            vol=total_vol,
            active=True
        )
        
        self.order_blocks.append(ob)
        logger.info(f"Created Bearish OB: top={ob.top:.2f} btm={ob.btm:.2f} strength={bearish_str/total_vol:.2%}")
        # Mark pivot as used so we don't create duplicate OBs from the same pivot
        try:
            self._used_pivots.add(pivot_idx)
        except Exception:
            # Defensive: if pivot_idx isn't available for any reason, continue
            pass
        
        # Generate signal (may be skipped if strength ratio too low)
        self.generate_signal("short", candles[breakout_idx], ob)
    
    
    def create_bullish_ob(self, candles: List[Candle], breakout_idx: int, pivot_idx: int):
        """
        Create a bullish order block after bearish breakout.
        Per new2_testing: Select lowest RED/BEARISH candle (close < open) before breakout.
        Choose by lowest LOW (not close).
        """
        # Select candidate candle from the bars immediately BEFORE the breakout
        # Match Latest_test.py semantics: search t-1 .. t-SWING_LENGTH (most recent swing_length bars)
        best_candle_idx = None
        lowest_low = float('inf')
        for j in range(1, self.SWING_LENGTH + 1):
            idx = breakout_idx - j
            if idx < 0:
                break
            c = candles[idx]
            if c.close < c.open and c.low < lowest_low:
                lowest_low = c.low
                best_candle_idx = idx

        # Fallback: choose the minimum low in the same backward window regardless of color
        if best_candle_idx is None:
            for j in range(1, self.SWING_LENGTH + 1):
                idx = breakout_idx - j
                if idx < 0:
                    break
                c = candles[idx]
                if c.low < lowest_low:
                    lowest_low = c.low
                    best_candle_idx = idx
        
        if best_candle_idx is None:
            return  # No suitable candle found
        
        ob_candle = candles[best_candle_idx]
        
        # Calculate strengths (use full SWING_LENGTH to match Latest_test behavior)
        bullish_str, bearish_str = self.calculate_strengths(candles, best_candle_idx, self.SWING_LENGTH)
        total_vol = bullish_str + bearish_str
        
        # Use MIN_TOTAL_VOLUME as a floor for ratio calculation, not as a gate
        # Always create OB even if volume is low (per Latest_test.py behavior)
        if total_vol < self.MIN_TOTAL_VOLUME:
            total_vol = self.MIN_TOTAL_VOLUME
        
        # OB geometry: use window around selected candle (matches Latest_test.py)
        half_window = self.ob_search_window // 2
        ws = max(0, best_candle_idx - half_window)
        we = min(len(candles) - 1, best_candle_idx + half_window)
        ob_top = max(c.high for c in candles[ws:we+1])
        ob_btm = min(c.low for c in candles[ws:we+1])
        
        # Create OB (always created, regardless of volume)
        ob = OrderBlock(
            kind="bullish",
            top=ob_top,
            btm=ob_btm,
            start_time=ob_candle.time,
            create_time=candles[breakout_idx].time,
            bullish_str=bullish_str,
            bearish_str=bearish_str,
            vol=total_vol,
            active=True
        )
        
        self.order_blocks.append(ob)
        logger.info(f"Created Bullish OB: top={ob.top:.2f} btm={ob.btm:.2f} strength={bullish_str/total_vol:.2%}")
        # Mark pivot as used so we don't create duplicate OBs from the same pivot
        try:
            self._used_pivots.add(pivot_idx)
        except Exception:
            pass

        # Generate signal (may be skipped if strength ratio too low)
        self.generate_signal("long", candles[breakout_idx], ob)
    
    
    def calculate_strengths(self, candles: List[Candle], selected_idx: int, bars_to_consider: int) -> tuple:
        """
        Sum volume backwards from selected_idx for bars_to_consider bars.
        Bearish volume: open > close, Bullish volume: close >= open
        """
        bullish = 0.0
        bearish = 0.0
        start = max(0, selected_idx - (bars_to_consider - 1))
        
        for i in range(selected_idx, start - 1, -1):
            c = candles[i]
            if c.open > c.close:
                bearish += c.volume
            else:
                bullish += c.volume
        
        return bullish, bearish
    
    
    def calculate_atr(self, candles: List[Candle], end_idx: int) -> float:
        """
        Calculate ATR based on ATR_METHOD.
        'highlow_sma': average of (high - low) over period
        'maxmin_range': (max(high) - min(low)) * range_percent over period (FULL structural range, not per-bar average)
        'close_range': (max(close) - min(close)) * range_percent over period (FULL structural range, not per-bar average)
        
        Per new2_testing: maxmin_range and close_range are full structural ranges, NOT divided by period.
        """
        if end_idx < self.atr_period:
            return 0.0
        
        start = end_idx - self.atr_period + 1
        
        if self.ATR_METHOD == 'highlow_sma':
            ranges = [candles[i].high - candles[i].low for i in range(start, end_idx + 1)]
            return np.mean(ranges)
        
        elif self.ATR_METHOD == 'maxmin_range':
            highs = [candles[i].high for i in range(start, end_idx + 1)]
            lows = [candles[i].low for i in range(start, end_idx + 1)]
            struct_range = (max(highs) - min(lows)) * self.RANGE_PERCENT
            return struct_range  # Return full structural range, NOT divided by period
        
        elif self.ATR_METHOD == 'close_range':
            closes = [candles[i].close for i in range(start, end_idx + 1)]
            close_range = (max(closes) - min(closes)) * self.RANGE_PERCENT
            return close_range  # Return full structural range, NOT divided by period
        
        return 0.0
    
    
    def check_ob_violations(self):
        """
        Check if any active OBs have been violated by recent price action.
        Violation rules based on VIOLATION_TYPE.
        """
        if not self.candles:
            return
        
        last_candle = self.candles[-1]
        
        for ob in self.order_blocks:
            if not ob.active:
                continue
            
            violated = False
            
            if self.VIOLATION_TYPE == "Close":
                # Violated if close price crosses through OB
                if ob.kind == "bullish" and last_candle.close < ob.btm:
                    violated = True
                elif ob.kind == "bearish" and last_candle.close > ob.top:
                    violated = True
            
            elif self.VIOLATION_TYPE == "Wick":
                # Violated if any price (high/low) crosses through OB
                if ob.kind == "bullish" and last_candle.low < ob.btm:
                    violated = True
                elif ob.kind == "bearish" and last_candle.high > ob.top:
                    violated = True
            
            if violated:
                ob.active = False
                ob.violated_time = last_candle.time
                logger.info(f"{ob.kind.capitalize()} OB violated: {ob.btm:.2f}-{ob.top:.2f}")
            else:
                # Reinforcement: CONSERVATIVE body-only test (matching Latest_test.py)
                # Only reinforce if the candle BODY (not wicks) lies fully inside the OB.
                # This avoids counting touching/violating wicks and prevents reinforcement
                # on bars that actually violate the OB.
                try:
                    body_low = min(last_candle.open, last_candle.close)
                    body_high = max(last_candle.open, last_candle.close)
                    body_inside = (body_low >= ob.btm) and (body_high <= ob.top)
                    
                    if ob.active and body_inside and last_candle.volume > 0:
                        # Classify by candle body direction
                        if last_candle.close >= last_candle.open:
                            ob.bullish_str += float(last_candle.volume)
                        else:
                            ob.bearish_str += float(last_candle.volume)
                        ob.vol = ob.bullish_str + ob.bearish_str
                        logger.debug(f"Reinforcement added {last_candle.volume} vol to OB ({ob.kind}) at {last_candle.time}: bull={ob.bullish_str:.4f} bear={ob.bearish_str:.4f}")
                except Exception:
                    # Defensive: don't allow reinforcement logic to break OB processing
                    pass
    
    
    # =========================
    # Signal Generation & Execution
    # =========================
    def generate_signal(self, signal_type: str, candle: Candle, ob: OrderBlock):
        """
        Generate a trading signal based on OB strength and market conditions.
        Implements all filters from new2_testing.py including EMA, sideways detection, etc.
        """
        # Check if max loss limit has been reached
        if self.max_loss_reached:
            logger.warning(f"⚠️ Skipping signal: Max loss limit reached (Cumulative PnL: ${self.cumulative_pnl:.2f})")
            return
        
        # Check strength ratio: use MIN_TOTAL_VOLUME as a floor for denominator
        # Prefer OB's stored total vol (created with MIN_TOTAL_VOLUME floor) as denominator.
        # Fall back to raw sum if ob.vol missing, then enforce floor — do NOT return early.
        total_vol = getattr(ob, 'vol', None)
        if total_vol is None or total_vol <= 0:
            total_vol = ob.bullish_str + ob.bearish_str
        total_vol = max(total_vol, self.MIN_TOTAL_VOLUME)
        
        if signal_type == "long":
            ratio = ob.bullish_str / total_vol
            if ratio < self.MIN_STRENGTH_RATIO:
                logger.info(f"Skipping long signal: strength ratio {ratio:.2%} < {self.MIN_STRENGTH_RATIO:.2%}")
                return
        
        elif signal_type == "short":
            ratio = ob.bearish_str / total_vol
            if ratio < self.MIN_STRENGTH_RATIO:
                logger.info(f"Skipping short signal: strength ratio {ratio:.2%} < {self.MIN_STRENGTH_RATIO:.2%}")
                return
        
        # Check max concurrent positions
        if len(self.open_positions) >= self.MAX_CONCURRENT_POSITIONS:
            logger.info(f"Skipping signal: max concurrent positions ({self.MAX_CONCURRENT_POSITIONS}) reached")
            return
        
        # Calculate EMA for entry filter
        ema = self.calculate_ema(len(self.candles) - 1)
        
        # Entry price determination: prefer live quote data for immediate execution
        # ProfitView provides real-time bid/ask via quote_update callbacks
        if self.ENTRY_PRICE_MODE.lower() == "close":
            # Use candle close, but prefer live quote midpoint if available
            if self.current_bid is not None and self.current_ask is not None:
                entry_price = (self.current_bid + self.current_ask) / 2
            else:
                entry_price = candle.close
        else:  # "worst"
            # Use worst case: ask for longs (more expensive), bid for shorts (less favorable)
            if signal_type == "long":
                entry_price = self.current_ask if self.current_ask is not None else candle.high
            else:
                entry_price = self.current_bid if self.current_bid is not None else candle.low
        
        # EMA filter
        if signal_type == "long":
            ema_diff = (entry_price - ema) / ema
            if ema_diff > self.ENTRY_DIFF_LONG_PCT:
                logger.info(f"Skipping long: price {ema_diff:.2%} above EMA")
                return
        else:  # short
            ema_diff = (ema - entry_price) / ema
            if ema_diff > self.ENTRY_DIFF_SHORT_PCT:
                logger.info(f"Skipping short: price {ema_diff:.2%} below EMA")
                return
        
        # Sideways market detection
        candle_list = list(self.candles)
        current_idx = len(candle_list) - 1
        atr = self.calculate_atr(candle_list, current_idx)
        atr_pct = atr / entry_price if entry_price > 0 else 0
        
        capital_multiplier = 1.0
        
        if atr_pct < self.ATR_SIDEWAYS_THRESHOLD_PCT:
            if self.SIDEWAYS_ACTION == 'skip':
                logger.info(f"Skipping signal: sideways market (ATR {atr_pct:.3%} < {self.ATR_SIDEWAYS_THRESHOLD_PCT:.3%})")
                return
            elif self.SIDEWAYS_ACTION == 'leverage':
                capital_multiplier = self.SIDEWAYS_LEVERAGE_MULT
                logger.info(f"Sideways market: applying {capital_multiplier}x capital multiplier")
        
        # Calculate position size using entry-vs-OB distance (matches Latest_test.py)
        ob_size = ob.top - ob.btm
        
        # Base distance from entry to OB boundary
        if signal_type == "long":
            # For longs: SL below entry, measure distance to OB bottom
            raw_dist = max(0.0, entry_price - ob.btm)
        else:
            # For shorts: SL above entry, measure distance to OB top
            raw_dist = max(0.0, ob.top - entry_price)
        
        # Apply ATR and percentage floors to base distance
        atr_floor_val = atr * self.OB_MIN_SL_ATR_MULT
        pct_floor = entry_price * self.OB_MIN_SL_PCT
        base_dist = max(raw_dist, atr_floor_val, pct_floor)
        
        # Scale by multiplier to get final SL distance
        sl_distance = base_dist * self.STOP_LOSS_MULTIPLIER
        
        # Per new2_testing: Add position sizing risk floors to prevent huge positions when SL is tiny
        min_risk_floor = entry_price * self.MIN_RISK_PCT  # e.g., 0.2% of entry price
        atr_half_floor = atr * self.ATR_HALF_FLOOR  # e.g., 0.5 * ATR
        risk_per_unit = max(sl_distance, min_risk_floor, atr_half_floor)
        
        # Guard against zero or negative sl_distance (Fix Issue #5)
        if sl_distance <= 0:
            logger.warning(f"Skipping signal: invalid sl_distance={sl_distance:.6f} (OB size={ob_size:.6f}, ATR={atr:.6f})")
            return
        
        # Calculate SL and TP prices
        if signal_type == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * self.TAKE_PROFIT_MULTIPLIER)
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * self.TAKE_PROFIT_MULTIPLIER)
        
        # Position sizing based on risk
        # Guard against None capital (Fix Issue #6)
        capital_for_sizing = self.initial_capital if self.USE_FIXED_CAPITAL else self.current_capital
        
        if capital_for_sizing is None:
            logger.warning(f"Capital not yet initialized, using fallback: ${self.FALLBACK_CAPITAL}")
            capital_for_sizing = self.FALLBACK_CAPITAL
            # Also set initial_capital if not set
            if self.initial_capital is None:
                self.initial_capital = self.FALLBACK_CAPITAL
            if self.current_capital is None:
                self.current_capital = self.FALLBACK_CAPITAL
        
        capital_for_sizing *= capital_multiplier
        
        risk_amount = capital_for_sizing * (self.RISK_PER_TRADE_PERCENT / 100)
        position_size = risk_amount / (risk_per_unit if risk_per_unit > 0 else 1.0)
        
        # Apply max position size constraint
        position_value = position_size * entry_price
        if position_value > self.MAX_POSITION_SIZE_USD:
            position_size = self.MAX_POSITION_SIZE_USD / entry_price
            logger.info(f"Position size capped at ${self.MAX_POSITION_SIZE_USD}")
        
        # Create position object
        position = TradingPosition(
            entry_time=candle.time,
            entry_price=entry_price,
            position_type=signal_type,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ob_size=ob_size,
            sl_distance=sl_distance,
            tp_distance=sl_distance * self.TAKE_PROFIT_MULTIPLIER,
            capital_at_risk=risk_amount,
            initial_stop_loss=stop_loss,
            highest_price=entry_price if signal_type == "long" else None,
            lowest_price=entry_price if signal_type == "short" else None,
            holding_counter=0  # Start at 0 for new position
        )
        
        # Reset holding_counter for same-side positions (reinforcing signal)
        for pos in self.open_positions:
            if pos.position_type == signal_type:
                pos.holding_counter = 0
        
        logger.info(f"=== SIGNAL GENERATED ===")
        logger.info(f"Type: {signal_type.upper()} | Entry: {entry_price:.2f} | Size: {position_size:.4f}")
        logger.info(f"SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Risk: ${risk_amount:.2f}")
        
        # Execute order
        self.execute_order(position, "Buy" if signal_type == "long" else "Sell")
    
    
    def calculate_ema(self, end_idx: int) -> float:
        """
        Calculate EMA for entry filter.
        Per new2_testing: Uses exponential weighted mean (ewm), not simple mean.
        """
        if end_idx < self.ema_period:
            return 0.0
        
        candle_list = list(self.candles)
        start = end_idx - self.ema_period + 1
        
        # Extract closes for the window
        closes = [candle_list[i].close for i in range(start, end_idx + 1)]
        
        # Calculate EMA using exponential weighting (span = period)
        # EMA formula: alpha = 2 / (span + 1), then iteratively weight
        alpha = 2.0 / (self.ema_period + 1)
        ema = closes[0]  # Start with first value
        
        for close in closes[1:]:
            ema = alpha * close + (1 - alpha) * ema
        
        return ema
    
    
    def execute_order(self, position: TradingPosition, side: str):
        """
        Execute market order to open position.
        Per ProfitView docs: create_market_order returns {'data': {'order_id': str, ...}}
        Handles immediate fills (Fix Issue #2)
        """
        try:
            resp = self.create_market_order(
                self.VENUE,
                self.SYMBOL,
                side=side,
                size=position.position_size
            )
            
            if resp and not resp.get('error'):
                order_data = resp.get('data', {})
                order_id = order_data.get('order_id')
                
                position.order_id = order_id
                
                # Try to extract actual fill price from create_market_order response
                fill_px = (order_data.get('fill_price') or 
                          order_data.get('order_price') or 
                          order_data.get('price'))
                if fill_px and position.actual_entry_price is None:
                    try:
                        position.actual_entry_price = float(fill_px)
                        position.entry_filled_size = position.position_size
                        logger.info(f"Set actual_entry_price from order response: {position.actual_entry_price:.2f}")
                    except (ValueError, TypeError):
                        pass
                
                # Track pending order or handle immediate fill
                if order_id:
                    # Some venues return an order_id even when the market order is
                    # immediately filled. Detect that case and treat as executed
                    # (move to open_positions) instead of leaving in pending_orders.
                    remain = order_data.get('remain_size')
                    status = (order_data.get('status') or '').lower()
                    filled_size = (order_data.get('filled_size') or order_data.get('fill_size') or order_data.get('filled') or 0)
                    try:
                        filled_qty = float(filled_size)
                    except Exception:
                        filled_qty = 0.0

                    already_filled = (remain == 0) or (status in ('filled', 'closed')) or (filled_qty >= (position.position_size - 1e-9))

                    if already_filled:
                        # Treat as immediate fill
                        logger.info(f"✓ Market order executed immediately (order_id given): {side} {position.position_size:.4f} | Order ID: {order_id}")
                        self.open_positions.append(position)
                        # Update fee delta after immediate fill
                        self._update_fee_delta()
                    else:
                        self.pending_orders[order_id] = position
                        logger.info(f"✓ Market order submitted: {side} {position.position_size:.4f} | Order ID: {order_id}")
                else:
                    # Immediately filled - no order_id (Fix Issue #2)
                    fill_price = (order_data.get('fill_price') or 
                                 order_data.get('fillPrice') or 
                                 order_data.get('price') or 
                                 order_data.get('avg_price'))
                    
                    if fill_price:
                        try:
                            position.actual_entry_price = float(fill_price)
                            position.entry_filled_size = position.position_size
                            logger.info(f"✓ Market order filled immediately: {side} {position.position_size:.4f} @ {fill_price}")
                        except (ValueError, TypeError):
                            logger.warning(f"Could not parse immediate fill price: {fill_price}")
                    else:
                        logger.info(f"✓ Market order filled immediately: {side} {position.position_size:.4f} (no fill price in response)")
                    
                    # Add to open positions immediately
                    self.open_positions.append(position)
                    # Update fee delta after immediate fill
                    self._update_fee_delta()
                
                self.positions.append(position)
                self.stats['total_trades'] += 1
            
            else:
                error = resp.get('error') if resp else 'No response'
                logger.error(f"Failed to execute order: {error}")
        
        except Exception as e:
            logger.exception(f"Error executing order: {e}")
    
    
    # =========================
    # Position Management
    # =========================
    def check_exit_conditions(self):
        """
        Check if any open positions should be closed.
        Checks SL, TP, and holding period.
        """
        if not self.current_price:
            return
        
        positions_to_close = []
        
        for position in self.open_positions:
            should_close = False
            exit_reason = None
            
            # Check stop loss
            if position.position_type == "long":
                if self.current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
            else:  # short
                if self.current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "stop_loss"
            
            # Check take profit
            if position.position_type == "long":
                if self.current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
            else:  # short
                if self.current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "take_profit"
            
            # Check holding period (use counter, not index lookup - Issue #6 fix)
            if self.HOLDING_PERIOD_BARS > 0:
                if position.holding_counter >= self.HOLDING_PERIOD_BARS:
                    should_close = True
                    exit_reason = "holding_period"
            
            if should_close:
                positions_to_close.append((position, exit_reason))
        
        # Close positions
        for position, reason in positions_to_close:
            self.close_position(position, self.current_price, reason)
    
    
    def check_trailing_stops(self):
        """
        Update and check trailing stops for open positions.
        Implements the full trailing stop logic from Latest_test.py including:
        - ATR-aware trailing stop levels
        - Progressive SL updates when price moves
        - Buffer candles before allowing exit
        """
        if not self.current_price:
            return
        
        current_time = self.epoch_now
        candle_list = list(self.candles)
        current_idx = len(candle_list) - 1
        
        # Calculate ATR for trailing stop floor
        atr = self.calculate_atr(candle_list, current_idx) if candle_list else 0.0
        
        for position in self.open_positions:
            # Update peak prices with None-safe comparisons
            if position.position_type == "long":
                if position.highest_price is None:
                    position.highest_price = self.current_price
                elif self.current_price > position.highest_price:
                    position.highest_price = self.current_price
            else:  # short
                if position.lowest_price is None:
                    position.lowest_price = self.current_price
                elif self.current_price < position.lowest_price:
                    position.lowest_price = self.current_price
            
            # Check if trailing should be activated
            if not position.trailing_active:
                if position.position_type == "long":
                    if position.highest_price is not None:
                        profit = position.highest_price - position.entry_price
                        activation_threshold = position.sl_distance * self.TRAILING_STOP_ACTIVATION
                        
                        if profit >= activation_threshold:
                            position.trailing_active = True
                            position.trailing_activation_time = current_time
                            position.trailing_activation_idx = self._get_candle_index_for_time(current_time)
                            position.last_trailing_update_price = position.highest_price
                            
                            # Compute ATR-aware trailing stop level on activation
                            pct_level = position.highest_price * (1 - self.TRAILING_STOP_PERCENT / 100)
                            atr_level = position.highest_price - (self.TRAILING_ATR_MULT * atr)
                            new_trailing_sl = max(pct_level, atr_level)
                            
                            # Move SL up immediately (only if higher than current SL)
                            prev_sl = position.stop_loss
                            position.stop_loss = max(position.stop_loss, new_trailing_sl)
                            
                            logger.info(f"Trailing stop activated for long @ {self.current_price:.2f} | SL {prev_sl:.2f} -> {position.stop_loss:.2f}")
                
                else:  # short
                    if position.lowest_price is not None:
                        profit = position.entry_price - position.lowest_price
                        activation_threshold = position.sl_distance * self.TRAILING_STOP_ACTIVATION
                        
                        if profit >= activation_threshold:
                            position.trailing_active = True
                            position.trailing_activation_time = current_time
                            position.trailing_activation_idx = self._get_candle_index_for_time(current_time)
                            position.last_trailing_update_price = position.lowest_price
                            
                            # Compute ATR-aware trailing stop level on activation
                            pct_level = position.lowest_price * (1 + self.TRAILING_STOP_PERCENT / 100)
                            atr_level = position.lowest_price + (self.TRAILING_ATR_MULT * atr)
                            new_trailing_sl = min(pct_level, atr_level)
                            
                            # Move SL down immediately (only if lower than current SL)
                            prev_sl = position.stop_loss
                            position.stop_loss = min(position.stop_loss, new_trailing_sl)
                            
                            logger.info(f"Trailing stop activated for short @ {self.current_price:.2f} | SL {prev_sl:.2f} -> {position.stop_loss:.2f}")
            
            # Update trailing stop if active
            if position.trailing_active:
                current_candle_idx = self._get_candle_index_for_time(current_time)
                candles_since_activation = current_candle_idx - position.trailing_activation_idx if position.trailing_activation_idx is not None else 999
                
                # Update SL if buffer allows and price moved enough
                if candles_since_activation >= self.TRAILING_STOP_BUFFER_CANDLES:
                    if position.position_type == "long":
                        # Compute new ATR-aware trailing level
                        pct_level = position.highest_price * (1 - self.TRAILING_STOP_PERCENT / 100)
                        atr_level = position.highest_price - (self.TRAILING_ATR_MULT * atr)
                        new_trailing_sl = max(pct_level, atr_level)
                        
                        # Only update if price moved enough since last update
                        if position.last_trailing_update_price is not None:
                            price_move_pct = ((position.highest_price - position.last_trailing_update_price) / position.last_trailing_update_price) * 100
                            
                            if price_move_pct >= self.TRAILING_STOP_UPDATE_THRESHOLD:
                                prev_sl = position.stop_loss
                                position.stop_loss = max(position.stop_loss, new_trailing_sl)
                                position.last_trailing_update_price = position.highest_price
                                logger.info(f"Trailing SL updated for long: {prev_sl:.2f} -> {position.stop_loss:.2f} (price moved {price_move_pct:.2f}%)")
                        
                        # Check if trailing SL hit
                        if self.current_price <= position.stop_loss:
                            logger.info(f"Trailing stop hit for long: price {self.current_price:.2f} <= SL {position.stop_loss:.2f}")
                            self.close_position(position, position.stop_loss, "trailing_stop")
                    
                    else:  # short
                        # Compute new ATR-aware trailing level
                        pct_level = position.lowest_price * (1 + self.TRAILING_STOP_PERCENT / 100)
                        atr_level = position.lowest_price + (self.TRAILING_ATR_MULT * atr)
                        new_trailing_sl = min(pct_level, atr_level)
                        
                        # Only update if price moved enough since last update
                        if position.last_trailing_update_price is not None:
                            price_move_pct = ((position.last_trailing_update_price - position.lowest_price) / position.last_trailing_update_price) * 100
                            
                            if price_move_pct >= self.TRAILING_STOP_UPDATE_THRESHOLD:
                                prev_sl = position.stop_loss
                                position.stop_loss = min(position.stop_loss, new_trailing_sl)
                                position.last_trailing_update_price = position.lowest_price
                                logger.info(f"Trailing SL updated for short: {prev_sl:.2f} -> {position.stop_loss:.2f} (price moved {price_move_pct:.2f}%)")
                        
                        # Check if trailing SL hit
                        if self.current_price >= position.stop_loss:
                            logger.info(f"Trailing stop hit for short: price {self.current_price:.2f} >= SL {position.stop_loss:.2f}")
                            self.close_position(position, position.stop_loss, "trailing_stop")
    
    
    def close_position(self, position: TradingPosition, exit_price: float, exit_reason: str):
        """
        Close a position via market order.
        Per ProfitView docs: use create_market_order with opposite side.
        
        Uses actual fill prices when available, falls back to current price estimate.
        """
        try:
            # Determine closing side (opposite of opening)
            side = "Sell" if position.position_type == "long" else "Buy"
            
            # Mark position as closing so fill_update knows to track exit fills
            position.is_closing = True
            
            logger.info(f"Closing {position.position_type} position: {side} {position.position_size:.4f} @ {exit_price:.2f} | Reason: {exit_reason}")
            
            # Capture fee snapshot before close
            fee_before_close = self.last_known_fees
            
            resp = self.create_market_order(
                self.VENUE,
                self.SYMBOL,
                side=side,
                size=position.position_size
            )
            
            if resp and not resp.get('error'):
                # Update fee delta to capture exit fees
                self._update_fee_delta()
                
                # Calculate exit fee delta (difference from before close)
                exit_fee_delta = self.last_known_fees - fee_before_close
                if exit_fee_delta > 1e-6:
                    position.exit_fees_paid = exit_fee_delta
                    logger.info(f"Exit fee delta captured: {exit_fee_delta:.4f}")
                
                # Update position with exit info
                position.exit_time = self.epoch_now
                position.exit_reason = exit_reason
                
                # Use actual fill price if available, otherwise use provided exit_price
                final_exit_price = position.actual_exit_price if position.actual_exit_price else exit_price
                position.exit_price = final_exit_price
                
                # Use actual entry price if available
                final_entry_price = position.actual_entry_price if position.actual_entry_price else position.entry_price
                
                # Calculate PnL using actual prices
                if position.position_type == "long":
                    position.pnl = final_exit_price - final_entry_price
                else:  # short
                    position.pnl = final_entry_price - final_exit_price
                
                position.pnl_percent = (position.pnl / final_entry_price) * 100 if final_entry_price != 0 else 0
                
                # Calculate total fees from entry + exit
                total_fees = position.entry_fees_paid + position.exit_fees_paid
                
                if total_fees > 0:
                    logger.info(f"Using actual fees from delta tracking: ${total_fees:.4f} (entry: ${position.entry_fees_paid:.4f}, exit: ${position.exit_fees_paid:.4f})")
                else:
                    # Fallback estimate if delta tracking failed
                    estimated_commission = (final_entry_price + final_exit_price) * (self.COMMISSION_PERCENT / 100) * position.position_size
                    total_fees = estimated_commission
                    logger.warning(f"Using estimated commission (delta tracking failed): ${estimated_commission:.4f}")
                
                position.pnl_dollars = (position.pnl * position.position_size) - total_fees
                
                # Update cumulative PnL and check loss limit
                self.cumulative_pnl += position.pnl_dollars
                
                # Calculate dynamic loss limit: base limit + earned profits
                # If we've earned $1000, we can lose up to $6000 total before pausing
                dynamic_loss_limit = -(self.MAX_LOSS_LIMIT - max(0, self.cumulative_pnl))
                
                if self.cumulative_pnl <= dynamic_loss_limit and not self.max_loss_reached:
                    self.max_loss_reached = True
                    logger.critical(f"⚠️ MAX LOSS LIMIT REACHED! Cumulative PnL: ${self.cumulative_pnl:.2f} | Limit: ${dynamic_loss_limit:.2f}")
                    logger.critical(f"🛑 TRADING PAUSED - Use /resume endpoint to re-enable (careful!)")
                    self.running = False
                
                # Update stats
                self.stats['total_pnl'] += position.pnl_dollars
                if position.pnl_dollars > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                # Update capital if not using fixed capital
                if not self.USE_FIXED_CAPITAL:
                    self.current_capital += position.pnl_dollars
                
                # Log with actual vs estimated price info
                price_source = "actual" if position.actual_exit_price else "estimated"
                logger.info(f"✓ Position closed: PnL ${position.pnl_dollars:.2f} ({position.pnl_percent:.2f}%) | Exit price: {price_source}")
                logger.info(f"Account stats: Cumulative PnL ${self.cumulative_pnl:.2f} | Total PnL ${self.stats['total_pnl']:.2f} | W/L {self.stats['winning_trades']}/{self.stats['losing_trades']}")
                
                # Remove from open positions
                self.open_positions.remove(position)
            
            else:
                error = resp.get('error') if resp else 'No response'
                logger.error(f"Failed to close position: {error}")
                position.is_closing = False  # Reset flag on failure
        
        except Exception as e:
            logger.exception(f"Error closing position: {e}")
            position.is_closing = False  # Reset flag on exception
    
    
    def _get_candle_index_for_time(self, time_ms: int) -> int:
        """
        Find candle index for a given timestamp.
        Returns nearest candle index to avoid returning 0 on mismatch (Issue #4 fix).
        """
        if not self.candles:
            return 0
        
        # Try exact match first
        for i, candle in enumerate(self.candles):
            if candle.time == time_ms:
                return i
        
        # No exact match - find nearest candle by time
        min_diff = float('inf')
        nearest_idx = 0
        
        for i, candle in enumerate(self.candles):
            diff = abs(candle.time - time_ms)
            if diff < min_diff:
                min_diff = diff
                nearest_idx = i
        
        return nearest_idx
    
    
    # =========================
    # HTTP Webhooks (per ProfitView docs)
    # =========================
    @http.route
    def post_pause(self, data):
        """Pause the trading strategy"""
        self.running = False
        logger.info("🛑 Strategy PAUSED")
        return {"status": "paused", "message": "Strategy execution paused"}
    
    
    @http.route
    def post_resume(self, data):
        """Resume the trading strategy"""
        self.running = True
        # Allow resuming even if max loss was reached (user must be careful!)
        if self.max_loss_reached:
            logger.warning("⚠️ Resuming despite max loss limit reached - proceed with caution!")
        logger.info("▶️ Strategy RESUMED")
        return {"status": "running", "message": "Strategy execution resumed"}
    
    
    @http.route
    def post_update_params(self, data):
        """
        Update trading parameters at runtime.
        
        Accepts parameters as JSON body:
        {
            "risk_per_trade_percent": 2.0,
            "max_concurrent_positions": 5,
            "stop_loss_multiplier": 1.5,
            "take_profit_multiplier": 20.0,
            "trailing_stop_activation": 2.5,
            "trailing_stop_percent": 1.5,
            "trailing_stop_buffer_candles": 15,
            "max_loss_limit": 7000.0,
            "min_strength_ratio": 0.35,
            "swing_length": 35,
            "holding_period_bars": 600
        }
        
        Returns updated parameters and confirmation message.
        """
        updated = []
        errors = []
        
        # Validate and update parameters
        param_mappings = {
            # Risk & Capital Management
            "risk_per_trade_percent": ("RISK_PER_TRADE_PERCENT", float, 0.1, 10.0),
            "max_concurrent_positions": ("MAX_CONCURRENT_POSITIONS", int, 1, 20),
            "max_position_size_usd": ("MAX_POSITION_SIZE_USD", float, 100, 1000000),
            "max_loss_limit": ("MAX_LOSS_LIMIT", float, 100, 100000),
            
            # Position Management
            "stop_loss_multiplier": ("STOP_LOSS_MULTIPLIER", float, 0.5, 5.0),
            "take_profit_multiplier": ("TAKE_PROFIT_MULTIPLIER", float, 1.0, 50.0),
            "holding_period_bars": ("HOLDING_PERIOD_BARS", int, 0, 2000),
            
            # Trailing Stops
            "trailing_stop_activation": ("TRAILING_STOP_ACTIVATION", float, 1.0, 10.0),
            "trailing_stop_percent": ("TRAILING_STOP_PERCENT", float, 0.1, 10.0),
            "trailing_stop_buffer_candles": ("TRAILING_STOP_BUFFER_CANDLES", int, 0, 100),
            "trailing_stop_update_threshold": ("TRAILING_STOP_UPDATE_THRESHOLD", float, 0.0, 5.0),
            
            # Order Block Detection
            "swing_length": ("SWING_LENGTH", int, 5, 100),
            "min_strength_ratio": ("MIN_STRENGTH_RATIO", float, 0.1, 0.9),
            "break_atr_mult": ("BREAK_ATR_MULT", float, 0.0, 1.0),
            "ob_min_sl_atr_mult": ("OB_MIN_SL_ATR_MULT", float, 0.0, 10.0),
            "ob_min_sl_pct": ("OB_MIN_SL_PCT", float, 0.0, 0.1),
            
            # Entry Filters
            "entry_diff_long_pct": ("ENTRY_DIFF_LONG_PCT", float, 0.0, 0.2),
            "entry_diff_short_pct": ("ENTRY_DIFF_SHORT_PCT", float, 0.0, 0.2),
            "atr_sideways_threshold_pct": ("ATR_SIDEWAYS_THRESHOLD_PCT", float, 0.0, 0.01),
            "sideways_leverage_mult": ("SIDEWAYS_LEVERAGE_MULT", float, 0.1, 10.0),
        }
        
        for key, value in data.items():
            if key not in param_mappings:
                errors.append(f"Unknown parameter: {key}")
                continue
            
            attr_name, expected_type, min_val, max_val = param_mappings[key]
            
            try:
                # Type conversion
                converted_value = expected_type(value)
                
                # Range validation
                if not (min_val <= converted_value <= max_val):
                    errors.append(f"{key}: value {converted_value} out of range [{min_val}, {max_val}]")
                    continue
                
                # Update the attribute
                old_value = getattr(self, attr_name)
                setattr(self, attr_name, converted_value)
                updated.append(f"{key}: {old_value} → {converted_value}")
                
                # Special handling for computed parameters
                if key == "swing_length":
                    self.ob_search_window = max(1, int(self.SWING_LENGTH * self.OB_SEARCH_WINDOW_MULT))
                    self.atr_period = max(1, int(self.SWING_LENGTH * self.ATR_PERIOD_MULT))
                    self.ema_period = max(1, int(self.SWING_LENGTH * self.EMA_PERIOD_MULT))
                    self.atr_sideways_window = max(1, int(self.SWING_LENGTH * self.ATR_SIDEWAYS_WINDOW_MULT))
                    updated.append(f"  → ob_search_window: {self.ob_search_window}")
                    updated.append(f"  → atr_period: {self.atr_period}")
                    updated.append(f"  → ema_period: {self.ema_period}")
                
                # Reset max loss flag if limit is changed
                if key == "max_loss_limit" and self.max_loss_reached:
                    logger.info("Max loss limit updated - resetting max_loss_reached flag")
                    self.max_loss_reached = False
            
           
            
            except (ValueError, TypeError) as e:
                errors.append(f"{key}: invalid value type - {e}")
        
        logger.info(f"✓ Parameters updated via webhook: {len(updated)} changes")
        for u in updated:
            logger.info(f"  {u}")
        
        if errors:
            logger.warning(f"Parameter update errors: {errors}")
        
        return {
            "status": "success" if updated else "error",
            "updated": updated,
            "errors": errors,
            "message": f"Updated {len(updated)} parameter(s)" if updated else "No valid parameters updated"
        }
    
    
    @http.route
    def get_status(self, data):
        """Get current strategy status and statistics"""
        # Calculate dynamic loss limit
        dynamic_loss_limit = -(self.MAX_LOSS_LIMIT - max(0, self.cumulative_pnl))
        loss_buffer_remaining = self.cumulative_pnl - dynamic_loss_limit
        
        # Fetch live balance for accurate capital (fixes stale value issue)
        live_capital = self.current_capital  # fallback to cached
        try:
            balance_resp = self.fetch_balances(self.VENUE)
            if balance_resp and not balance_resp.get('error'):
                balances = balance_resp.get('data', [])
                for bal in balances:
                    asset = bal.get('asset', '').upper()
                    if asset in ['USD', 'USDT', 'USDC', 'BUSD']:
                        amount = float(bal.get('amount', 0))
                        if amount > 0:
                            live_capital = amount
                            break
        except Exception as e:
            logger.warning(f"get_status: failed to fetch live balance: {e}")
        
        return {
            "running": self.running,
            "initialized": self.initialized,
            "capital": live_capital,
            "capital_cached": self.current_capital,  # show both for comparison
            "initial_capital": self.initial_capital,
            "cumulative_pnl": self.cumulative_pnl,
            "max_loss_limit": self.MAX_LOSS_LIMIT,
            "dynamic_loss_limit": dynamic_loss_limit,
            "loss_buffer_remaining": loss_buffer_remaining,
            "max_loss_reached": self.max_loss_reached,
            "open_positions": len(self.open_positions),
            "pending_orders": len(self.pending_orders),
            "active_order_blocks": len([ob for ob in self.order_blocks if ob.active]),
            "total_order_blocks": len(self.order_blocks),
            "candles_loaded": len(self.candles),
            "pivot_highs": len(self.pivot_highs),
            "pivot_lows": len(self.pivot_lows),
            "stats": self.stats,
            "parameters": {
                # Trading Configuration
                "symbol": self.SYMBOL,
                "venue": self.VENUE,
                "candle_level": self.CANDLE_LEVEL,
                
                # Order Block Detection
                "swing_length": self.SWING_LENGTH,
                "violation_type": self.VIOLATION_TYPE,
                "min_strength_ratio": self.MIN_STRENGTH_RATIO,
                "break_atr_mult": self.BREAK_ATR_MULT,
                "ob_search_window": self.ob_search_window,
                
                # Position Management
                "stop_loss_multiplier": self.STOP_LOSS_MULTIPLIER,
                "take_profit_multiplier": self.TAKE_PROFIT_MULTIPLIER,
                "max_concurrent_positions": self.MAX_CONCURRENT_POSITIONS,
                
                # Trailing Stops
                "trailing_stop_activation": self.TRAILING_STOP_ACTIVATION,
                "trailing_stop_percent": self.TRAILING_STOP_PERCENT,
                "trailing_stop_buffer_candles": self.TRAILING_STOP_BUFFER_CANDLES,
                
                # Capital Management
                "risk_per_trade_percent": self.RISK_PER_TRADE_PERCENT,
                "use_fixed_capital": self.USE_FIXED_CAPITAL,
                "max_position_size_usd": self.MAX_POSITION_SIZE_USD,
                "max_loss_limit": self.MAX_LOSS_LIMIT,
                "commission_percent": self.COMMISSION_PERCENT,
                "slippage_percent": self.SLIPPAGE_PERCENT,
                
                # Entry Filters
                "entry_price_mode": self.ENTRY_PRICE_MODE,
                "ema_period": self.ema_period,
                "entry_diff_long_pct": self.ENTRY_DIFF_LONG_PCT,
                "entry_diff_short_pct": self.ENTRY_DIFF_SHORT_PCT,
                
                # ATR Settings
                "atr_method": self.ATR_METHOD,
                "atr_period": self.atr_period,
                "ob_min_sl_atr_mult": self.OB_MIN_SL_ATR_MULT,
                "ob_min_sl_pct": self.OB_MIN_SL_PCT,
                
                # Sideways Market
                "atr_sideways_threshold_pct": self.ATR_SIDEWAYS_THRESHOLD_PCT,
                "sideways_action": self.SIDEWAYS_ACTION,
                "sideways_leverage_mult": self.SIDEWAYS_LEVERAGE_MULT,
                
                # Other
                "holding_period_bars": self.HOLDING_PERIOD_BARS
            }
        }
    
    
    @http.route
    def get_positions(self, data):
        """Get all open and closed positions with full details"""
        return {
            "open_positions": [
                {
                    "type": p.position_type,
                    "entry_time": p.entry_time,
                    "entry_price": p.entry_price,
                    "actual_entry_price": p.actual_entry_price,
                    "entry_fees_paid": p.entry_fees_paid,
                    "size": p.position_size,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_active": p.trailing_active,
                    "highest_price": p.highest_price,
                    "lowest_price": p.lowest_price,
                    "ob_size": p.ob_size,
                    "sl_distance": p.sl_distance,
                    "tp_distance": p.tp_distance,
                    "capital_at_risk": p.capital_at_risk,
                    "order_id": p.order_id,
                    "initial_stop_loss": p.initial_stop_loss
                }
                for p in self.open_positions
            ],
            "closed_positions": [
                {
                    "type": p.position_type,
                    "entry_time": p.entry_time,
                    "entry_price": p.entry_price,
                    "actual_entry_price": p.actual_entry_price,
                    "entry_fees_paid": p.entry_fees_paid,
                    "exit_time": p.exit_time,
                    "exit_price": p.exit_price,
                    "actual_exit_price": p.actual_exit_price,
                    "exit_fees_paid": p.exit_fees_paid,
                    "exit_reason": p.exit_reason,
                    "size": p.position_size,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "pnl": p.pnl,
                    "pnl_percent": p.pnl_percent,
                    "pnl_dollars": p.pnl_dollars,
                    "ob_size": p.ob_size,
                    "sl_distance": p.sl_distance,
                    "tp_distance": p.tp_distance,
                    "capital_at_risk": p.capital_at_risk,
                    "trailing_active": p.trailing_active,
                    "highest_price": p.highest_price,
                    "lowest_price": p.lowest_price,
                    "initial_stop_loss": p.initial_stop_loss,
                    "used_actual_prices": p.actual_entry_price is not None and p.actual_exit_price is not None
                }
                for p in self.positions if p.exit_time is not None
            ],
            "all_positions_summary": [
                {
                    "type": p.position_type,
                    "entry_time": p.entry_time,
                    "entry_price": p.entry_price,
                    "actual_entry_price": p.actual_entry_price,
                    "size": p.position_size,
                    "exit_time": p.exit_time,
                    "exit_price": p.exit_price,
                    "actual_exit_price": p.actual_exit_price,
                    "exit_reason": p.exit_reason,
                    "pnl_dollars": p.pnl_dollars,
                    "entry_fees": p.entry_fees_paid,
                    "exit_fees": p.exit_fees_paid,
                    "total_fees": p.entry_fees_paid + p.exit_fees_paid if p.exit_time else p.entry_fees_paid,
                    "used_actual_prices": p.actual_entry_price is not None and (p.actual_exit_price is not None if p.exit_time else True),
                    "status": "closed" if p.exit_time else "open"
                }
                for p in self.positions
            ],
            "summary": {
                "total_positions": len(self.positions),
                "open_count": len(self.open_positions),
                "closed_count": len([p for p in self.positions if p.exit_time is not None]),
                "pending_orders": len(self.pending_orders)
            },
            "fee_tracking": {
                "fee_baseline": self.fee_baseline,
                "last_known_fees": self.last_known_fees,
                "total_fees_paid": self.total_fees_paid,
                "stats_total_fees": self.stats['total_fees']
            },
            "parameters": {
                "symbol": self.SYMBOL,
                "venue": self.VENUE,
                "candle_level": self.CANDLE_LEVEL,
                
                # Position Management
                "risk_per_trade_percent": self.RISK_PER_TRADE_PERCENT,
                "use_fixed_capital": self.USE_FIXED_CAPITAL,
                "max_position_size_usd": self.MAX_POSITION_SIZE_USD,
                "stop_loss_multiplier": self.STOP_LOSS_MULTIPLIER,
                "take_profit_multiplier": self.TAKE_PROFIT_MULTIPLIER,
                "max_concurrent_positions": self.MAX_CONCURRENT_POSITIONS,
                
                # Trailing Stops
                "trailing_stop_activation": self.TRAILING_STOP_ACTIVATION,
                "trailing_stop_percent": self.TRAILING_STOP_PERCENT,
                "trailing_stop_buffer_candles": self.TRAILING_STOP_BUFFER_CANDLES,
                
                # Entry Filters
                "entry_price_mode": self.ENTRY_PRICE_MODE,
                "ema_period": self.ema_period,
                "entry_diff_long_pct": self.ENTRY_DIFF_LONG_PCT,
                "entry_diff_short_pct": self.ENTRY_DIFF_SHORT_PCT,
                
                # Capital & Fees
                "commission_percent": self.COMMISSION_PERCENT,
                "slippage_percent": self.SLIPPAGE_PERCENT,
                "current_capital": self.current_capital,
                "initial_capital": self.initial_capital,
                
                # Other
                "holding_period_bars": self.HOLDING_PERIOD_BARS
            }
        }
    
    
    @http.route
    def post_close_all(self, data):
        """Manually close all ETH positions"""
        logger.info("Manual close all ETH positions requested")
        
        closed_count = 0
        
        try:
            resp = self.fetch_positions(self.VENUE)
            
            if resp and not resp.get('error'):
                positions = resp.get('data', [])
                
                # Filter to only ETH positions
                eth_positions = [p for p in positions if p['sym'] == self.SYMBOL]
                
                for pos in eth_positions:
                    side = 'Buy' if pos['side'] == 'Sell' else 'Sell'
                    size = pos['pos_size']
                    
                    logger.info(f"Manually closing {pos['side']} ETH position: {side} {size}")
                    self.create_market_order(self.VENUE, self.SYMBOL, side=side, size=size)
                    closed_count += 1
                
                # Also update internal tracking
                self.open_positions.clear()
        except Exception as e:
            logger.exception(f"Error closing positions: {e}")
        
        return {
            "status": "success",
            "message": f"Closed {closed_count} ETH position(s)",
            "closed_count": closed_count
        }
    
    
    @http.route
    def get_orderblocks(self, data):
        """Get information about detected order blocks"""
        active_obs = [ob for ob in self.order_blocks if ob.active]
        violated_obs = [ob for ob in self.order_blocks if not ob.active]
        
        return {
            "active_order_blocks": [
                {
                    "kind": ob.kind,
                    "top": ob.top,
                    "btm": ob.btm,
                    "start_time": ob.start_time,
                    "create_time": ob.create_time,
                    "bullish_str": ob.bullish_str,
                    "bearish_str": ob.bearish_str,
                    "total_vol": ob.vol,
                    "strength_ratio": (ob.bullish_str / ob.vol if ob.kind == "bullish" else ob.bearish_str / ob.vol) if ob.vol > 0 else 0
                }
                for ob in active_obs
            ],
            "violated_order_blocks": [
                {
                    "kind": ob.kind,
                    "top": ob.top,
                    "btm": ob.btm,
                    "start_time": ob.start_time,
                    "create_time": ob.create_time,
                    "violated_time": ob.violated_time,
                    "bullish_str": ob.bullish_str,
                    "bearish_str": ob.bearish_str,
                    "total_vol": ob.vol
                }
                for ob in violated_obs[-10:]  # Last 10 violated OBs
            ],
            "summary": {
                "total_obs": len(self.order_blocks),
                "active_obs": len(active_obs),
                "violated_obs": len(violated_obs),
                "parameters": {
                    "swing_length": self.SWING_LENGTH,
                    "min_strength_ratio": self.MIN_STRENGTH_RATIO,
                    "violation_type": self.VIOLATION_TYPE,
                    "ob_search_window": self.ob_search_window
                }
            }
        }
    
    
    @http.route
    def get_market(self, data):
        """Get current market state and recent candles"""
        last_candles = list(self.candles)[-5:] if self.candles else []
        
        return {
            "current_price": self.current_price,
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "last_5_candles": [
                {
                    "time": c.time,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume
                }
                for c in last_candles
            ],
            "last_pivot_high": {
                "index": self.last_pivot_high[0] if self.last_pivot_high else None,
                "price": self.last_pivot_high[1] if self.last_pivot_high else None,
                "time": self.last_pivot_high[2] if self.last_pivot_high else None
            } if self.last_pivot_high else None,
            "last_pivot_low": {
                "index": self.last_pivot_low[0] if self.last_pivot_low else None,
                "price": self.last_pivot_low[1] if self.last_pivot_low else None,
                "time": self.last_pivot_low[2] if self.last_pivot_low else None
            } if self.last_pivot_low else None,
            "total_candles": len(self.candles),
            "parameters": {
                "candle_level": self.CANDLE_LEVEL,
                "swing_length": self.SWING_LENGTH
            }
        }
