# Crypto Causality

## Project Goal

1. Determine causality for data using Granger, Multivariate Granger, and/or Time-varying Granger Causality.
2. Determine appropriate lags for the data and appropriate causal structure (which causes which?).
3. Identify frequencies where causality is more pronounced or less pronounced.
4. Determine if the relationship is stable over time.
5. Assess if the relationship is asymmetric.
6. Use the findings to predict future prices/returns.
7. Develop a trading model based on this information.
8. Provide relevant statistics for the trading model.
9. Write a Notebook or report on your findings, linking to your code base.

## Granger Causality

Granger causality is a statistical hypothesis test to determine if one time series can predict another. It is based on the principle that if a time series X Granger-causes time series Y, then past values of X should contain information that helps predict Y beyond the information contained in past values of Y alone.

## Rolling Multivariate Causality

The `rolling_multivariate_causality.py` script performs rolling multivariate Granger causality analysis using a Vector Autoregression (VAR) model. It analyzes the causal relationships between multiple cryptocurrencies over a rolling window, allowing for the detection of time-varying causal structures.

## Project Structure

```
crypto_causality/
├── data/
│   ├── raw/                    # Store raw CSV files
│   └── processed/              # Store processed parquet files
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py       # Data download functionality
│   │   └── processor.py        # Data processing and cleaning
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outliers.py         # Outlier detection
│   │   ├── causality.py        # Granger causality analysis
│   │   └── stationarity.py     # Stationarity tests
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── causality_viz.py    # Visualization tools for causality analysis
│   │   └── plots.py            # General plotting functions
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py          # Helper functions
├── examples/
│   ├── run_analysis.py
│   └── visualize_results.py
├── notebooks/
│   └── crypto_exploration.ipynb
├── tests/
│   └── test_analysis.py
├── requirements.txt
└── README.md
```

## Installation

1. Run `pip install -r requirements.txt` to install the required packages.
2. Run `../make_dir_structure.sh` to create the project structure.
3. Edit `src/config.py` to set your desired configuration parameters.

## Downloading Data

To download data, use the `downloader.py` script. Ensure your `config.py` file is set up correctly:

```python
# src/config.py

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
# Data directories
RAW_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Trading pairs configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']

# Time parameters
INTERVAL = '1h'  # Data granularity
START_DATE = '2024-01-01'
END_DATE = '2024-12-01'

# API configuration
BASE_URL = 'https://api.binance.com'

# Analysis parameters
DEFAULT_WINDOW = 30  # Default window size for rolling calculations
MAX_LAGS = 10  # Maximum lags for causality analysis
SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (12, 8)
```

Run the downloader script:

```sh
python -m src.data.downloader
```

## Running Rolling Multivariate Causality Analysis

To run the rolling multivariate causality analysis, use the `rolling_multivariate_causality.py` script. The script accepts several parameters:

- `--data_file`: Path to the input data file (default: "../data/processed/").
- `--window_size`: Size of the rolling window (default: 300).
- `--max_lags`: Maximum number of lags for the VAR model (default: 10).
- `--sig_level`: Significance level for p-values (default: 0.05).
- `--db_file`: SQLite database file to save results (default: "../data/results.db").
- `--interval`: Interval for the data (default: "1m").
- `--fit_freq`: Frequency of fitting the model (default: 1).
- `--block_size`: Number of rows to process before saving to the database (default: 200).

Example usage:

```sh
python rolling_multivariate_causality.py --data_file "../data/processed/" --window_size 300 --max_lags 10 --sig_level 0.05 --db_file "../data/results.db" --interval "1m" --fit_freq 1 --block_size 200
```

This will run the rolling multivariate Granger causality analysis on the specified data and save the results to the specified SQLite database file.