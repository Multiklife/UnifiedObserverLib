import asyncio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unifiedobserverlib import refined_predict_btc, RealTimeDataFetcher
import aiohttp
import logging

logging.basicConfig(level=logging.DEBUG)

async def backtest_model(start_date, end_date, prediction_horizon):
    async with aiohttp.ClientSession() as session:
        historical_data = await RealTimeDataFetcher.fetch_btc_data(
            session, 
            timeframe='1h',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

    if not historical_data:
        logging.error("No historical data available. Exiting.")
        return

    logging.info(f"Fetched {len(historical_data)} data points")

    predictions = []
    actual_prices = []
    
    for i in range(0, len(historical_data) - prediction_horizon, prediction_horizon):
        current_data = historical_data[:i+prediction_horizon]
        try:
            forecast = await refined_predict_btc(current_data, prediction_horizon)
            predictions.extend(forecast)
            actual_prices.extend(historical_data[i+prediction_horizon:i+2*prediction_horizon])
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")

    if not predictions or not actual_prices:
        logging.error("No predictions or actual prices available. Exiting.")
        return

    mse = np.mean((np.array(predictions) - np.array(actual_prices))**2)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_prices)))

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('BTC Price: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    prediction_horizon = 24  # 24 часа

    asyncio.run(backtest_model(start_date, end_date, prediction_horizon))

