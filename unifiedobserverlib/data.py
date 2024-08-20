import asyncio
import aiohttp
import snscrape.modules.twitter as sntwitter

class RealTimeDataFetcher:
    @staticmethod
    async def fetch_btc_data(session, timeframe='1h', period='1mo'):
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={timeframe}&limit=1000"
        async with session.get(url) as response:
            data = await response.json()
            return [float(item[4]) for item in data]  # Close prices

    @staticmethod
    async def fetch_social_data(session, query, limit=100):
        tweets = []
        async for tweet in sntwitter.TwitterSearchScraper(f'{query} lang:en').get_items():
            if len(tweets) >= limit:
                break
            tweets.append(tweet.content)
        return tweets

    @staticmethod
    async def fetch_all_data(timeframe='1h', period='1mo'):
        async with aiohttp.ClientSession() as session:
            btc_data, social_data = await asyncio.gather(
                RealTimeDataFetcher.fetch_btc_data(session, timeframe, period),
                RealTimeDataFetcher.fetch_social_data(session, "bitcoin")
            )
        return btc_data, social_data
