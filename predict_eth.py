import asyncio
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from unifiedobserverlib import refined_predict_btc
from sklearn.metrics import mean_squared_error

async def predict_and_evaluate(symbol='ETH-USD', timeframe='1d', prediction_days=30):
    # Загружаем данные за последние 3 месяца
    data = yf.download(symbol, interval=timeframe, period='3mo')

    # Разделяем данные на исторические и тестовые
    train_data = data[:-prediction_days]
    test_data = data[-prediction_days:]

    # Строим предсказание на основе данных до последнего месяца
    predictions = await refined_predict_btc(timeframe, prediction_days)

    # Сравниваем предсказание с реальными данными
    real_prices = test_data['Close'].values
    predicted_prices = np.array(predictions)

    mse = mean_squared_error(real_prices, predicted_prices)
    print(f"Mean Squared Error of the prediction: {mse}")

    # Визуализируем результаты
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-2*prediction_days:], data['Close'].values[-2*prediction_days:], label='Historical Prices')
    plt.plot(test_data.index, predicted_prices, 'ro-', label='Predicted Prices')
    plt.title(f'ETH Price Prediction Evaluation ({timeframe})')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Запускаем предсказание и оценку
asyncio.run(predict_and_evaluate())
