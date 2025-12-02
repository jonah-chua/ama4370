"""Binance data collector utility.

Provides BinanceDataCollector.fetch_candles to paginate historical klines.

This module extracts the collector from `testing.py` so it can be reused/imported
cleanly.
"""
from datetime import datetime, timedelta
import time
from typing import Optional
import requests
import pandas as pd


class BinanceDataCollector:
    """Collect unlimited historical data from Binance."""

    def __init__(self, futures: bool = True):
        """
        Args:
            futures: True for futures (like PERP), False for spot
        """
        if futures:
            self.base_url = "https://fapi.binance.com/fapi/v1"
        else:
            self.base_url = "https://api.binance.com/api/v3"

        self.session = requests.Session()

    def fetch_candles(
        self, 
        symbol: str = 'BTCUSDT', 
        interval: str = '15m', 
        days: int = 30, 
        limit: int = 1500,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch historical candles from Binance with pagination.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candlestick interval (e.g., '1m', '5m', '15m', '1h', '1d')
            days: Number of days to fetch (used if start_date/end_date not provided)
            limit: Number of candles per request (max 1500)
            start_date: Start date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (optional)
            end_date: End date in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (optional, defaults to now)

        Returns a pandas DataFrame with columns: timestamp, datetime, open, high, low, close, volume
        or None if no data collected.
        """
        # Parse dates if provided
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_time = int(end_dt.timestamp() * 1000)
        else:
            end_time = int(datetime.now().timestamp() * 1000)
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            start_time = int(start_dt.timestamp() * 1000)
        else:
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_candles = []
        current_start = start_time

        batch = 0
        while current_start < end_time:
            batch += 1
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': limit
            }

            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code != 200:
                    # stop on HTTP error
                    break

                data = response.json()
                if not data:
                    break

                for candle in data:
                    all_candles.append({
                        'timestamp': candle[0],
                        'datetime': datetime.fromtimestamp(candle[0] / 1000),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })

                # next start is last candle timestamp + 1ms
                current_start = data[-1][0] + 1

                if len(data) < limit:
                    break

                time.sleep(0.1)

            except Exception:
                break

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles)
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Filter by date range if start_date was provided
        if start_date:
            cutoff_start = datetime.strptime(start_date.split()[0], '%Y-%m-%d') if ' ' in start_date else datetime.strptime(start_date, '%Y-%m-%d')
            df = df[df['datetime'] >= cutoff_start]
        else:
            cutoff_time = datetime.now() - timedelta(days=days)
            df = df[df['datetime'] >= cutoff_time]
        
        return df


__all__ = ["BinanceDataCollector"]
