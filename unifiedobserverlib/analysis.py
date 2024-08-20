import numpy as np
import pywt
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.sentiment import SentimentIntensityAnalyzer

class CycleAnalyzer:
    @staticmethod
    def detect_cycles(data):
        result = seasonal_decompose(data, model='additive', period=24)
        return result

    @staticmethod
    def wavelet_analysis(data):
        coeffs = pywt.wavedec(data, 'db1', level=5)
        return coeffs

class FractalAnalyzer:
    @staticmethod
    def hurst_exponent(data):
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    @staticmethod
    def fractal_dimension(data):
        return 2 - FractalAnalyzer.hurst_exponent(data)

class SocialMediaAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)['compound']
