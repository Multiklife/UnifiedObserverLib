from setuptools import setup, find_packages

setup(
    name="unifiedobserverlib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mpmath',
        'plotly',
        'scikit-learn',
        'tensorflow',
        'yfinance',
        'pywavelets',
        'statsmodels',
        'hmmlearn',
        'nltk',
        'aiohttp',
        'snscrape',
        'scikit-optimize'
    ],
)
