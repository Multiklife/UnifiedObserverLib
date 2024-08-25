import numpy as np
from .models import UnifiedObserver, RealityWaveFunction

async def refined_predict_btc(future_steps=24):
    uo = UnifiedObserver(0.5, 1.0)
    rwf = RealityWaveFunction(uo)
    
    predictions = []
    for i in range(future_steps):
        wave_value = rwf.evolve(i / future_steps)
        prediction = 100 * (1 + np.real(wave_value) * 0.01)  # Простой пример предсказания
        predictions.append(prediction)
    
    return predictions
