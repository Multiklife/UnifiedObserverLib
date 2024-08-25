import aiohttp
import logging
import json
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

class RealTimeDataFetcher:
    @staticmethod
    async def fetch_btc_data(session, timeframe='1h', start_date=None, end_date=None):
        try:
            if start_date and end_date:
                logging.debug(f"Fetching historical data from {start_date} to {end_date}")
                btc = yf.Ticker("BTC-USD")
                data = btc.history(start=start_date, end=end_date, interval=timeframe)
                return data['Close'].tolist()
            else:
                url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={timeframe}&limit=1000"
                logging.debug(f"Fetching real-time data from URL: {url}")
                async with session.get(url) as response:
                    if response.status != 200:
                        logging.error(f"HTTP error: {response.status}")
                        return []
                    data = await response.json()
                    if not isinstance(data, list):
                        logging.error(f"Unexpected data format: {type(data)}")
                        return []
                    return [float(item[4]) for item in data]  # Close prices
        except Exception as e:
            logging.error(f"Error in fetch_btc_data: {str(e)}")
            return []
