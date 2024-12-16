# src/config.py

# Trading pairs to analyze
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "SOLUSDT"
]

# Time parameters
INTERVAL = "1m"
START_DATE = "2024-01-01"
END_DATE = "2024-02-01"

# API configuration
BASE_URL = "https://api.binance.com"

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
