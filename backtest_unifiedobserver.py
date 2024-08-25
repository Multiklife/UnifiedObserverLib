import asyncio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from unifiedobserverlib import refined_predict_btc, RealTimeDataFetcher
import aiohttp
import logging
import json

logging.basicConfig(level=logging.DEBUG)

async def fetch_data(session):
    try:
        data = await RealTimeDataFetcher.fetch_btc_data(session)
        logging.debug(f"Raw data type: {type(data)}")
        logging.debug(f"Raw data (first 100 characters): {str(data)[:100]}")
        
        if isinstance(data, str):
            # Если data - строка, попробуем распарсить её как JSON
            data = json.loads(data)
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], (list, tuple)) and len(data[0]) > 4:
                return [float(item[4]) for item in data]  # Предполагаем, что цена закрытия находится в 5-м элементе
            elif isinstance(data[0], (int, float)):
                return data  # Данные уже в нужном формате
            else:
                logging.error(f"Unexpected data format in list: {data[:5]}...")
        elif isinstance(data, dict):
            logging.error(f"Data is a dictionary. Keys: {data.keys()}")
        else:
            logging.error(f"Unexpected data type: {type(data)}")
        
        return []
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        return []

async def backtest_model(start_date, end_date, prediction_horizon):
    async with aiohttp.ClientSession() as session:
        historical_data = await fetch_data(session)

    if not historical_data:
        logging.error("No historical data available. Exiting.")
        return

    logging.info(f"Fetched {len(historical_data)} data points")

    predictions = []
    actual_prices = []
    
    for i in range(0, len(historical_data) - prediction_horizon, prediction_horizon):
        current_data = historical_data[:i+1]
        try:
            forecast = await refined_predict_btc(future_steps=prediction_horizon)
            predictions.append(forecast[-1])
            actual_price = historical_data[i + prediction_horizon]
            actual_prices.append(actual_price)
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

