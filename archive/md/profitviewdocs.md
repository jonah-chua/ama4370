
Logo

    Home

Trading

    Overview
    Getting Started
    Base Class

Event Callbacks
Exchange API
Create Websocket Feeds
Create Webhooks

        Glossary

    Changelog

    » Trading

Trading Bots#
Overview#

The algorithmic trading platform is built using an event-driven architecture. Events from the market trigger callback functions. These callbacks can either be private or public updates from the market. Private events include updates to the resting limit orders you have placed in the market, or updates to your positions resulting from an order being filled. Public events include updates to the top-of-book bid and ask quotes, and trades that have occurred due to orders being matched.

A template for an algorithmic trading strategy is shown below. In this template the strategy will generate "signals" which indicate what the model believes to be the fair value of the market. In instances when the signal generated fair value differs significantly from the actual market price, the model will then trigger buy or sell orders. These orders can either be passive (limit) or aggressive (market) reflecting the urgency that the position is sought.

ProfitView provides an interface to develop algorithmic trading strategies using this framework. Each component in the above diagram have callback functions, and API methods available to allow one to write automated strategies that trade the market.

We will describe these component callbacks and methods in detail. Before starting, it is recommended that you are comfortable with Python 3.9, as development of a strategy requires writing in this language. If you are new to programming or are coming from another language we can recommend the following online CodeAcademy Course which covers the basics of Python 3.
Getting Started#

To get started you will need to sign up to ProfitView and be on either a Hobby ($29/mo), Active Trader ($59/mo), or Professional ($299/mo) plan. After joining a plan the Trading Bots interface will become available, with a Python code editor.

When in the code editor create a new file and will notice a template script is provided which includes the following class:

class Trading(Link)

Note: trading bot strategies are unable to run if this class is not defined in your file.

We describe below various event-driven callbacks and the API reference for this class.
Base Class#

The base class Link has a number of helper properties and methods available.
class Link#

    now → datetime.datetime#

            Current UTC time

    unix_now → int#

            Current unix time i.e. seconds since midnight UTC on 1 Jan 1970

    epoch_now → int#

            Current the epoch time i.e. milliseconds since midnight UTC on 1 Jan 1970

    second → float#

            Current second (elapsed from start of minute) in microsecond precision

    iso_now → str#

            Current UTC time in ISO 8601 format

    background_threads → dict#

            Live background threads as dict with key "thread id" and value "thread name"

    candle_bin(value:int, level:int, ceil:bool=False) → int#

            Takes an epoch timestamp and time level e.g. '1m'. Returns the timestamp rounded to that time level. By default the timestamp is rounded down (ceil = False) to the the time level. It can also be rounded up (ceil = True).

            Example

>>> value = 1696016546768 # 2023-09-29T19:42:26.768000
>>> self.candle_bin(value, '15m', ceil=False)
1696015800000 # 2023-09-29T19:30:00

    time_range(start:int, end:int, level:str, include_start:bool=True) → list#

            Takes a start and end epoch timestamp with a time level. Returns a time range of epoch timestamps ranging from the start to the end times each differing by the time level, optionally including the start value.

            Example

>>> start = 1696015800000 # 2023-09-29T19:30:00
>>> end = 1696021200000 # 2023-09-29T21:00:00
>>> self.time_range(start, end, '15m', include_start=False)
[1696016700000, 1696017600000, 1696018500000, 1696019400000, 1696020300000, 1696021200000]

    create_thread(target, *args, **kwargs) → None#

            Helper function that allows one to create a thread, passing in the target function to be called and any of its args and kwargs. It is essentially shorthand for:

import threading
thread = threading.Thread(target, args=args, kwargs=kwargs)
thread.start()

    delay_thread(interval, target, *args, **kwargs) → None#

            Helper function that allows one to delay a thread by some interval seconds (float). This is shorthand for:

import threading
thread = threading.Timer(interval, target, args=args, kwargs=kwargs)
thread.start()

Event Callbacks#

Private updates include an update on the order placed on an exchange or confirmation that an order has been filled. These updates are used to manage risk and order placement logic.

Public updates include trades and top-of-book quote market activity for the symbols that have been subscribed to for the strategy that is currently running. These updates are used to generate and update the trading signals relating to a strategy.

Each callback is a function with the following

PARAMETERS:

    src: exchange identifier e.g. bitmex. See glossary for supported exchanges
    sym: symbol of event update XBTUSD
    data: event data depending on particular callback

Order Update (private)#

Receive private order updates from all connected exchanges.

def order_update(self, src:str, sym:str, data:dict)

Example data

{
    'venue': 'BitMEX',          # name of your venue
    'side': 'Buy',              # can be 'Buy' or 'Sell'
    'order_price': 20500.0,     # price of order placed
    'order_size': 1000.0,       # submitted size of order placed
    'remain_size': 1000.0,      # remaining size of order placed
    'order_type': 'LIMIT',      # typically 'LIMIT' or 'MARKET'
    'time': 1678320346997,      # epoch timestamp from exchange
    'order_id': 'd9a4ec1e8861', # unique order id from exchange
    'info': { object }          # unparsed data from exchange ws feed
}

Notes

    venue is the name set in your exchange settings
    order_id should be used when amending or cancelling orders with the Exchange API

Fill Update (private)#

Receive private trade fill updates from all connected exchanges.

def fill_update(self, src:str, sym:str, data:dict)

Example data

{
    'venue': 'BitMEX'           # name of your venue
    'side': 'Sell',             # can be 'Buy' or 'Sell'
    'fill_price': 21638.5,      # price order filled at
    'fill_size': 100.0,         # total filled size of order
    'order_type': 'MARKET',     # typically 'LIMIT' or 'MARKET'
    'time': 1678361696643,      # epoch timestamp from exchange
    'order_id': 'b0883c17a23b', # unique order id from exchange
    'trade_id': '67fd1a6c7069', # unique trade id from exchange
    'info': { object }          # unparsed data from exchange ws feed
}

Position Update (private)#

Receive private position updates from all connected exchanges.

def position_update(self, src:str, sym:str, data:dict)

Example data

{
    'venue': 'BitMEX'           # name of your venue
    'pos_size': -500.0,         # current position size of sym
    'mark_price': 64928.63,     # exchange mark price used for margining
    'entry_price': 65291.12,    # entry price of position if available
    'liq_price': 28870.5,       # liquidation price of position
    'time': 1713640060717,      # epoch timestamp from exchange
    'info': { object }          # unparsed data from exchange ws feed
}

Quote Update (public)#

Receive public market top-of-book quote updates for all subscribed symbols.

def quote_update(self, src:str, sym:str, data:dict)

Example data

{
    'bid': [21601, 170000],     # bid price, bid size
    'ask': [21601.5, 223000],   # ask price, ask size
    'time': 1678364572949       # epoch timestamp from exchange
}

Trade Update (public)#

Receive public market trade (orders which matched on the exchange) updates for all subscribed symbols.

def trade_update(self, src:str, sym:str, data:dict)

Example data

{
    'side': 'Buy',              # side of taker in trade
    'price': 21611.5,           # price of matched order
    'size': 200.0,              # size of matched order
    'time': 1678364683876       # epoch timestamp from exchange
}

Exchange API#

Trading bots interact with their connected exchange accounts using a common API. This includes: getting candle data, open orders, current positions; creating limit and market orders; cancelling and amending exisiting orders.

Each Exchange API call will return a dict object with the following

PARAMETERS:

    src: exchange identifier - see glossary for supported exchanges
    venue: name of your connected exchange as set in exchange settings
    data: payload response from calling exchange endpoint (either list or dict)
    error: exchange API request error
        returns None if no error
        example error:

{
    "type": "api_error",
    "message": "sym XBTZ23 expired"
}

    rate_limits: remaining exchange API rate limits
        foo

Rate Limits#

This dict object will contain the following keys:

    remaining: the remaining exchange API credits; if this number gets too low, the trading bot automatically backs off exchange requests to avoid a 429 Too Many Requests error.
    reset: the unix time in seconds when the rate limits will be reset; like remaining this is set by the exchange

Example

"rate_limits": {
    "remaining": 42,
    "reset": 1681187280
},

Error#

If this is not None it will be a dict object containing the following keys:

    type: the type of API error; this can be one of:
        "api_error": there was an error with the request payload or exchange; an exception will be thrown in your trading bot in this instance
        "rate_limits": too many exchange API requests; handle this by retrying your request after the reset value specified in rate_limits; no exception will be thrown in your trading bot
    message: more information about the error to help with debugging

Example

"error": {
    "type": "api_error",
    "message": "The symbol 'DOGGUSD' does not exist"
}

Fetch Candles#

self.fetch_candles(venue, sym=sym, level='1m', since=None)

Fetch open, high, low, close, volume (OHLCV) candle data for symbol from an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
sym 	str 		Symbol for which candle data is required
level 	str 		Time frame for candle data e.g. "1m", "15m", "1h", "1d"
since 	int 		Timestamp in milliseconds of the earliest candle to fetch

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": [
        {
            "open": 30140.0,
            "high": 30138.5,
            "low": 30124.5,
            "close": 30132.0,
            "volume": 100300.0,
            "time": 1681187280000,
        },
    ],
    "rate_limits": {
        "remaining": 56,
        "reset": 1681187280
    },
}

Param 	Type 	Description
open 	float 	Open price for time interval
high 	float 	Highest price in the time interval
low 	float 	Lowest price in the time interval
close 	float 	Close price for time interval
volume 	float 	Total volume traded in the time interval
time 	int 	Unix time of beginning of time interval
Fetch Balances#

self.fetch_balances(venue)

Fetch all wallet balances for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": [
        {
            "asset": "USDT",
            "amount": 4250.0
        },
        {
            "asset": "BMEX",
            "amount": 10.5
        },
        {
            "asset": "BTC",
            "amount": 0.56283
        }
    ],
    "rate_limits": {
        "remaining": 56,
        "reset": 1681187280
    },
}

Param 	Type 	Description
asset 	str 	Asset symbol of wallet balance
amount 	float 	Wallet balance amount
Fetch Open Orders#

self.fetch_open_orders(venue)

Fetch all open orders for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": [
        {
            "sym": "XBTUSD",
            "side": "Buy",
            "order_price": 18650.0,
            "order_size": 800.0,
            "remain_size": 300.0,
            "order_type": "LIMIT",
            "time": 1681264318158,
            "order_id": "46f383f8-4a94-4b47-8b0a-e6a5869bf201",
        }
    ],
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264328
    },
}

Param 	Type 	Description
sym 	str 	Symbol of open order
side 	str 	Side of order, can be either "Buy" or "Sell" only
order_price 	float 	Price at which an order has been placed
order_size 	float 	Submitted size of currently open order
remain_size 	float 	Remaining size of currently open order
order_type 	str 	Type of order, typically "LIMIT"
time 	int 	Integer unix time: number of milliseconds since the epoch 1 Jan 1970
order_id 	str 	Unique order id provided by exchange used to make order updates
Fetch Open Positions#

self.fetch_positions(venue)

Fetch all open positions for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": [
        {
            "sym": "XBTUSD",
            "side": "Buy",
            "entry_price": 21638.5,
            "liq_price": 8453.0,
            "pos_size": 100.0,
            "time": 1681264314158
        }
    ],
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Param 	Type 	Description
sym 	str 	Symbol of open position
side 	str 	Side of open position, can be either "Buy" or "Sell" only
entry_price 	float 	Average price at which a position has been entered into
liq_price 	float 	Liquidation price, only returned for swap, futures, margin positions
pos_size 	float 	Size of current position for the given instrument
time 	int 	Unix timestamp for which the position data was returned
Create Limit Order#

self.create_limit_order(venue, sym=sym, side=side, size=size, price=price)

Create a limit order for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
sym 	str 		Symbol of order to be submitted
side 	str 		Side of order, can be either "Buy" or "Sell" only
size 	float 		Size of order to be placed
price 	float 		Price of order to be placed

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": {
        "sym": "XBTUSD",
        "side": "Buy",
        "order_price": 25600.0,
        "order_size": 200.0,
        "remain_size": 200.0,
        "order_type": "LIMIT",
        "time": 1681262857139,
        "order_id": "f34afdeb-0c14-4864-9838-7bd914c59b98",
    },
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Param 	Type 	Description
sym 	str 	Symbol of order that has been placed
side 	str 	Side of order that has been placed
order_price 	float 	Price of order that has been placed
order_size 	float 	Order size of submitted order
remain_size 	float 	Remaining size of a submitted order
order_type 	str 	Type of order placed, only "LIMIT" orders are supported
time 	int 	Unix timestamp the order was entered into the order book
order_id 	str 	Unique order id of order that has been placed
Create Market Order#

self.create_market_order(venue, sym=sym, side=side, size=size)

Create a market order for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
sym 	str 		Symbol of order to be submitted
side 	str 		Side of order, can be either "Buy" or "Sell" only
size 	float 		Size of order to be placed
price 	float 		Price of order to be placed

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": {
        "sym": "XBTUSD",
        "side": "Sell",
        "order_price": 30002.5,
        "order_size": 100.0,
        "remain_size": 0.0,
        "order_type": "MARKET",
        "time": 1681267174064,
        "order_id": "2c56f74d-5fa4-3022-a2ec-89909e8f7ef3",
    },
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Param 	Type 	Description
sym 	str 	Symbol of order that has been placed
side 	str 	Side of order that has been placed
order_price 	float 	Price of order that has been placed by market
order_size 	float 	Order size of submitted order
remain_size 	float 	Remaining size of a submitted order
order_type 	str 	Type of order placed, only "MARKET" order type returned
time 	int 	Unix timestamp the order was entered into the order book
order_id 	str 	Unique order id of order that has been placed
Cancel Order#

self.cancel_order(venue, order_id=order_id, sym=sym)

Cancel a single or multiple open orders for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
order_id 	str 		ID of order to be cancelled
sym 	str 		Symbol of order to be submitted

Notes:

    If neither order_id or sym are provided all open orders will be cancelled
    If only order_id is provided, then this particular order will be cancelled
    If only sym is provided, then all orders with this symbol will be cancelled

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": [
        {
            "sym": "XBTUSD",
            "side": "Buy",
            "order_price": 29684.5,
            "order_size": 100.0,
            "remain_size": 0.0,
            "order_type": "LIMIT",
            "time": 1681263047320,
            "order_id": "c00b2035-3a25-47a8-95e3-852a2c92d37a"
        }
    ],
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Param 	Type 	Description
sym 	str 	Symbol of order that has been cancelled
side 	str 	Side of order that has been cancelled
order_price 	float 	Price of order that has been cancelled
order_size 	float 	Order size of cancelled order
remain_size 	float 	Remaining size of cancelled order
order_type 	str 	Type of order cancelled
time 	int 	Unix timestamp the order was cancelled
order_id 	str 	Unique order id of order of cancelled order
Amend Order#

self.amend_order(venue, order_id=order_id, size=size, price=price)

Amend a single open order for an exchange account.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
order_id 	str 		Order id of order to be amended
size 	str 		New size of order to be amended
price 	float 		New price of order to be amended

Note: at least one of price or size needs to be provided for the order amendment to be valid.

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "error": None,
    "data": {
        "sym": "XBTUSD",
        "side": "Buy",
        "order_price": 27875.5,
        "order_size": 100.0,
        "remain_size": 100.0,
        "order_type": "LIMIT",
        "time": 1681268424914,
        "order_id": "3da7e21a-35ce-482a-9b3e-2e424dafb8cc",
    },
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Param 	Type 	Description
sym 	str 	Symbol of order that has amended
side 	str 	Side of order that has been amended
order_price 	float 	Price of order that has been amended
order_size 	float 	Order size of amended order
remain_size 	float 	Remaining size of amended order
order_type 	str 	Type of order amended
time 	int 	Unix timestamp the order was amended
order_id 	str 	Unique order id of order of amended order
Call Endpoint#

self.call_endpoint(venue, path, version, method='GET', params=None)

Call a native REST API of the venue.
Param 	Type 	Required 	Description
venue 	str 		Name of exchange API key to make call with e.g. "BitMEX"
path 	str 		Path of the native API endpoint, e.g. "instrument"
version 	str 		"public" or "private"
method 	str 		e.g. "GET" or "POST"
params 	dict 		Dictionary of parameters to pass to the endpoint

Example Response

{
    "src": "bitmex",
    "venue": "BitMEX",
    "venue_id": "636f2042-c0e1-4d3e-95b9-cbf0dde58d5d",
    "error": None,
    "data": [{
        ...
        },
        {
        ...
        }, ...
    ]
    "rate_limits": {
        "remaining": 56,
        "reset": 1681264334
    },
}

Note: the data member will contain results that depend on the specific native API called.
Create Websocket Feeds#

Trading bots allow you to create your own private websocket feeds, that can be streamed out to other applications. To start streaming websocket messages, simply call:

self.publish(topic, data=None)

Publish message to private websocket stream.
Param 	Type 	Required 	Description
topic 	str 		Name of the websocket topic
data 	multiple 		Payload can be any JSON serializable object e.g. dict, list, etc

To connect to your private websocket feed use the url:

wss://profitview.net/stream?token=YOUR_API_KEY

Note: YOUR_API_KEY can be found in Account Settings.

Once connected you will receive messages in the following format

< { "type": your_topic, "data": your_data }

Example use cases:

    Streaming strategy state to a Grafana dashboard
    Stream signals to your local environment or another server for processing
    Build a web dashboard to view metrics of your strategy

Create Webhooks#

The trading bot comes with a built in HTTP server on which you can make GET and POST requests. To create an HTTP endpoint i.e. a webhook, each callback method needs to include the following decorator:

@http.route

Calling a webhook requires a WEBHOOK_SECRET which unique to your account. The list of registered webhook URLs (with the secret) can be found by clicking the bolt icon on the navigation panel of the code editor.
GET requests#

All GET requests must start with "get_" in the method name. For example the below webhook returns account balances for the queried exchange account:

@http.route
def get_balances(self, data):
    return self.fetch_balances(data['venue'])

This webhook can be accessed using cURL for example as follows:

curl https://profitview.net/trading/bot/WEBHOOK_SECRET/balances?venue=BitMEX

POST requests#

All POST requests must start with "post_" in the method name. For example the below webhook cancels the order provided in the payload:

@http.route
def post_cancel_order(self, data):
    return self.cancel_order(data['venue'], order_id=data['order_id'])

This webhook can be accessed using cURL for example as follows:

curl -X POST https://profitview.net/trading/bot/WEBHOOK_SECRET/cancel_order \
    -d 'venue=BitMEX' \
    -d 'order_id=ORDER_ID'

Example use cases:

    Send TradingView signals to trigger an order
    View and set the trading strategy state on a Google Sheet
    Use the HTTP REST interface in your local environment or another server

Glossary#
Enum Definitions#
level#

    1m, 3m, 5m, 15m, 30m - minute
    1h, 2h, 3h, 4h, 6h, 12h - hour
    1d - day
    1w - week

Installed Libraries#

Each trading instance comes pre-installed with the following popular libraries useful for algoritmic trading and technical analysis:

    numpy: high performance library for multidimension array and matrix calculations
    pandas: library for creating data structures and performing analysis
    scikit-learn: machine learning library for predictive analysis, built on NumPy and SciPy
    scipy: library for mathematical algorithms and convenience functions
    TA-Lib: popular trading software library to perform technical analysis of market data

If there is a Python library that is not listed here that you would like to see available, please request if by filling in the form here.
Supported Exchanges#
ID 	Name 	Spot 	Margin 	Swap 	Futures 	Options
bitmex 	BitMEX 					
coinbasepro 	Coinbase Pro 					
woo 	WOO X 					

More exchanges coming soon!
