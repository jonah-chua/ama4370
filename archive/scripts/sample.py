from profitview import Link, http, logger

# See docs: https://profitview.net/docs/trading/
class Trading(Link):
	'''
	A Simple SMA (Simple Moving Average) Crossover Trading Strategy
	This strategy employs two SMAs to identify short-term trends and momentum.

	The short-term SMA (leading indicator), calculated using the most recent 2 trade prices.
	The long-term SMA (lagging indicator), calculated using the most recent 5 trade prices.

	A trading signal is triggered when the short-term SMA crosses above or below the long-term SMA 
	by a margin exceeding the threshold. This signal determines whether to open a long (cross above)
	or short (cross below) position. 

	To simplify position management, the algorithm only opens one position at a time. 
	Once a position is opened, it monitors the current market price to apply either a 
	stop-profit or stop-loss for closing the open position.
	
	Note:
	This is an extremely short-term, high-frequency trading strategy, 
	ideal for demo purposes as it could generate trade actions quickly. 
	
	However, there are limitations:
		Using such a short look-back period may not adequately capture a stable trend.
		Excessive trading can lead to losses due to slippage, spread, and commission costs.
		Exceed the API calling limits in short time.
	'''
	
	
	SRC = 'woo'                 # Crypto Exchange
	VENUE = 'WooPaper'          # Trading Venue (Account)
	SYMBOL = 'PERP_BTC_USDT'    # Crypto Product

	# Define trading parameters	
	SMA_SHORT_TRADES = 2		# number of trades for the short term SMA calculation
	SMA_LONG_TRADES = 5			# number of trades for the long term SMA calculation
	THRESHOLD_PERCENT = 0.005	# the long/term SMA crossover needs to the more than the THRESHOLD_PERCENT in order to generate the long/short signal to open a position
	STOP_PROFIT_PERCENT = 0.01	# Stop profit%
	STOP_LOSS_PERCENT = -0.01	# Stop loss%

	# Position managment: for simplicitly only trade with 1 open position size
	hasPosition = False			# Whether the account has an open position
	ORDER_SIZE = 1 				# The position size
	
	# Real-time market related info
	tradePxList = []	        # The list to store the most recent 5 trade px to calculate the long and short term SMA
    stopProfitPx = 0	        # Stop Profit price
	stopLossPx = 0		        # Stop Loss price

	# Risk and FFL check
	MAX_BID_ASK_SPREAD_BPS = 1			# Do not trade when bid/ask spread is too wide. More liquid products usually has smaller spread 
	MAX_TRADE_NOTIONAL_USD = 1000000	# Max dollar notional per trade
	
	# (Optional) System control parameters
	init = True # for running the first time
    running = True # pause or resume the strategy
		
        
    #Called on every market trades from subscribed symbols
    def trade_update(self, src, sym, data):
        # (Optional system code) Only called for the first run to close all the existing positions if any
        if self.init and self.hasPositions():
            self.closePosition()
            
        self.init = False
		# One-time fetch of account info (balances / fee schedule) - best effort
		if not getattr(self, '_fetched_account_info', False):
			try:
				self.fetch_account_info()
			except Exception:
				logger.exception('trade_update: failed to fetch account info')
        
		''' The main trading logic starts from here'''
		if self.running:
			# Update with the latest market data
			tradePx = data['price']
			self.updateTradePxList(tradePx)
			
			# When there's no position, try to determine the open position
            if not self.hasPosition:
                self.determineOpenTrade()
            
            # When there is position, try to determine the stopLoss or stopProfit condition to close the opening position 
            else:
                self.determineStopTrade(tradePx)
        	

	# Update tradePxList with the last trade px
    def updateTradePxList(self, tradePx):
		# Update the tradePxList with the current tradePx
		self.tradePxList.append(tradePx)
        
        # Only use the most recent trade px defined in SMA_LONG_TRADES
        self.tradePxList = self.tradePxList[max(0, len(self.tradePxList) - self.SMA_LONG_TRADES): ]
		logger.info(self.tradePxList)


	def fill_update(self, src, sym, data):
		"""Record per-fill fees for accurate reporting (best-effort)."""
		# many platforms include 'fee' or 'fee_amount' in fill objects
		fee = data.get('fee') or data.get('fee_amount') or data.get('fee_value')
		if fee:
			try:
				if not hasattr(self, 'total_fees_paid'):
					self.total_fees_paid = 0.0
				self.total_fees_paid += float(fee)
				logger.info(f"fill_update: fee={fee} currency={data.get('fee_currency')} total_fees={self.total_fees_paid}")
			except Exception:
				logger.exception('fill_update: failed to record fee')
     

	# The main trade logic
	def determineOpenTrade(self):
        # Have enough number of trade px, do the trade logic
		if len(self.tradePxList) == self.SMA_LONG_TRADES:
            '''
            1. Calculate trading indicators (smaLong, smaShort) from the current tradePxList
                Using the recent SMA_LONG_TRADES number of trades for the smaLong (lagging) indicator
                Using the recent SMA_SHORT_TRADES number of trades for the smaShort (leading) indicator
            '''
            smaLong = sum(self.tradePxList) / self.SMA_LONG_TRADES 
            smaShort = sum(self.tradePxList[(self.SMA_LONG_TRADES - self.SMA_SHORT_TRADES):]) / self.SMA_SHORT_TRADES
            # Determine the trading signal and send order accordling

            '''
            2. Determine the trading signal
                 1 (long signal): when sma crossover is bigger than the threshold (short term uptrend)
                 0: (no action signal): when sma crossover is within the threshold (signal not strong enough yet) and don't do anything
                -1: (short signal): when sma crossover is smaller the threshold (short term downtrend)
            '''
            diffPercent = (smaShort - smaLong) / smaShort * 100
            
            if diffPercent > self.THRESHOLD_PERCENT: signal = 1
            elif diffPercent < -self.THRESHOLD_PERCENT: signal = -1
            else: signal = 0

            logger.info(f"smaLong={smaLong:.4f}, smaShort={smaShort:.4f}, threshold%={self.THRESHOLD_PERCENT}, diff%={diffPercent:.4f}, signal={signal}")
            
            '''
            3. Open a trade position if signal is non-zero
            '''
            if signal != 0:
                self.openPosition(signal)

		
	# Depends on the signal, open a Buy(long) / Sell(short) position		
	def openPosition(self, signal):
		if signal == 1 or signal == -1:
			# signal ==  1: Buy
            # signal == -1: Sell
            side = 'Buy' if signal == 1 else 'Sell'
			size = self.ORDER_SIZE
            
			logger.info(f"Opening {side} position")
			if self.riskCheck(side, size):
				self.sendMarketOrder(side, size)
				self.calcStopPrice()
		else:
  			logger.error('Wong signal')

			
	# Risk and FFL check
	# Cannot open a position if the risk check is failed
	def riskCheck(self, side, size):
		# get current market quote (bid/ask)
        quote = self.quotes[self.SRC][self.SYMBOL] 
		bid = quote['bid'][0]
		ask = quote['ask'][0]

		# check trade notional
		tradePx = ask if side == 'Buy' else bid
		tradeNotional = tradePx * size
		
		if tradeNotional > self.MAX_TRADE_NOTIONAL_USD:
			logger.warning(f"The trade notional ({tradeNotional}) exceel the max amount: {self.MAX_TRADE_NOTIONAL_USD}")
			return False
		
		# Bid Ask Spread check - do not trade when spread is too wide
		bidAskSpreadBps = ((ask - bid) / ((bid + ask) / 2)) * 10000

		if bidAskSpreadBps > self.MAX_BID_ASK_SPREAD_BPS:
			logger.warning(f"The bid/ask spread ({bidAskSpreadBps:.4f}) exceel the max spread: {self.MAX_BID_ASK_SPREAD_BPS}")
			return False
		
		# Pass all the risk check
		logger.info(f"OK: bid={bid}, ask={ask}, bidAskSpreadBps={bidAskSpreadBps:.4f}, tradeNotional: {tradeNotional}")
		return True

	
	# Send the market order execution instruction to the exchange to actually open/close a position
	def sendMarketOrder(self, side, size):
		# Send market order to the exchange
        ret = self.create_market_order(self.VENUE, self.SYMBOL, side=side, size=size)
		retData = ret['data']		

		# update the current position 
		self.hasPosition = self.hasPositions()
		logger.info(f"Market Order Executed: {retData['side']} {retData['order_size']}@{retData['order_price']}, hasPosition={self.hasPosition}")
			
	
	# Calculate stopProfit and stopLoss price
    def calcStopPrice(self):
        # Calculate the stop profit and stop loss after opening a position
		# It's based on the open position side and price
        positionSide = self.fetch_positions(self.VENUE)['data'][0]["side"]
        positionPx = self.fetch_positions(self.VENUE)['data'][0]["entry_price"]
        
		# Calculate the stopProfit and stopLoss px for the Long Position
        if positionSide == 'Buy':
            self.stopProfitPx = positionPx * (1 + self.STOP_PROFIT_PERCENT / 100)
            self.stopLossPx = positionPx * (1 - self.STOP_PROFIT_PERCENT / 100)
		
		# Calculate the stopProfit and stopLoss px for the Short Position
        else:
            self.stopProfitPx = positionPx * (1 - self.STOP_PROFIT_PERCENT / 100)
            self.stopLossPx = positionPx * (1 + self.STOP_PROFIT_PERCENT / 100)
            
        logger.info(f"stopProfitPx={self.stopProfitPx:.4f}, stopLossPx={self.stopLossPx:.4f}")    

    
	# Check whether to stopProfit or stopLoss given the current market trade price
	def determineStopTrade(self, tradePx):
        positionSide = self.fetch_positions(self.VENUE)['data'][0]["side"]
        logger.info(f"Check StopPrice for {positionSide} position: tradePx={tradePx}, stopProfitPx={self.stopProfitPx:.4f}, stopLossPx={self.stopLossPx:.4f}")
		
		'''
        For long position: 
			StopProfit	: if the current market tradePx > stopProfitPx
			StopLoss	: if the current market tradePx < stopLossPx

        For short position: 
			StopProfit	: if the current market tradePx < stopProfitPx
			StopLoss	: if the current market tradePx > stopLossPx
		'''
        if positionSide == 'Buy':
			if tradePx > self.stopProfitPx:
                logger.info("### StopProfit")
				self.closePosition()
			elif tradePx < self.stopLossPx:
				logger.info("### StopLoss")
				self.closePosition()

		elif positionSide == 'Sell':
			if tradePx < self.stopProfitPx:
				logger.info("### StopProfit")
				self.closePosition()
			elif tradePx > self.stopLossPx:
				logger.info("### StopLoss")
				self.closePosition()	

	
	# Check if the account currently has any opening position
    def hasPositions(self):
        positions = self.fetch_positions(self.VENUE)['data']
        
        if len(positions) > 0: return True
        else: return False
        

	# Close all the opening positions
	def closePosition(self):	
		positions = self.fetch_positions(self.VENUE)['data']

		# for each open position
		for position in positions:
			'''
			Send Buy order with position size to close the Sell position 
			Send Sell order with position size to close the Buy position
			'''        
			side = 'Buy' if position['side'] == 'Sell' else 'Sell'
			logger.info(f"Closing position: {side} {position['sym']}, qty={position['pos_size']} @ Market")
            self.sendMarketOrder(side, position['pos_size'])


	# --- Account / Fee helpers ---
	def fetch_account_info(self):
		"""Best-effort: populate account-related info (capital/fees) from the platform.
		This will not raise on failure; it logs and sets a fetched flag.
		"""
		if getattr(self, '_fetched_account_info', False):
			return
		try:
			resp = None
			# try common API names
			if hasattr(self, 'fetch_balances'):
				resp = self.fetch_balances(self.VENUE)
			elif hasattr(self, 'fetch_accounts'):
				resp = self.fetch_accounts()
			elif hasattr(self, 'fetch_account'):
				resp = self.fetch_account(self.VENUE)

			if not resp or (isinstance(resp, dict) and resp.get('error')):
				logger.debug('fetch_account_info: no account data or error')
				self._fetched_account_info = True
				return

			data = resp.get('data') if isinstance(resp, dict) else resp
			# Try to detect commission percent in account/venue metadata
			acc = None
			if isinstance(data, list) and data:
				# try to pick matching venue or first
				for a in data:
					if a.get('venue') == self.VENUE or a.get('account') == self.VENUE:
						acc = a
						break
				if acc is None:
					acc = data[0]
			elif isinstance(data, dict):
				acc = data

			if isinstance(acc, dict):
				for k in ('maker_fee_percent', 'taker_fee_percent', 'fee_percent', 'fee'):
					if k in acc and acc[k] is not None:
						try:
							self.COMMISSION_PERCENT = float(acc[k])
							logger.info(f'fetch_account_info: set COMMISSION_PERCENT={self.COMMISSION_PERCENT}')
							break
						except Exception:
							pass

		except Exception:
			logger.exception('fetch_account_info: unexpected error')
		finally:
			self._fetched_account_info = True


	'''
	The following are optional webhook function which allow ProfitView to communicate external
	'''
	
	# Pause the current strategy from running the main trade logic
	@http.route
	def get_pause(self, data):
		self.running = False
		logger.info("Strategy paused")
		return "Paused"

	
	# Resume the current strategy to run the main trade logic
	@http.route
	def get_resume(self, data):
		self.running = True
		logger.info("Strategy resumed")
		return "Resumed"
