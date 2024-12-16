# Crypto Causality

1. The Goal is to determine causal relationships between BTC and other major crypto currencies: ETH, BNB, XRP, ADA,DOGE,SOL
2. Which causes which? Is this relationship stable? Is it persistent over shorter horizons?
3. Can we get by with 1 min data? Is causality more pronounced at higher frequencies? 



---- 
## The Project structure

crypto_causality/
├── data/
│   ├── raw/                    # Store raw CSV files
│   └── processed/              # Store processed parquet files
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py      # Data download functionality
│   │   └── processor.py       # Data processing and cleaning
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outliers.py        # Outlier detection
│   │   └── causality.py       # Granger causality analysis
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Helper functions
├── notebooks/
│   └── analysis.ipynb         # Jupyter notebook for exploration
├── tests/
│   └── __init__.py
├── requirements.txt
└── README.md


## Revised Project Structure

crypto_analysis/
│
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── outliers.py
│   │   ├── causality.py 
│   │   └── stationarity.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── processor.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── causality_viz.py 
│   │   └── plots.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── examples/
│   ├── run_analysis.py
│   └── visualize_results.py
├── notebooks/
│   └── crypto_exploration.ipynb
├── tests/
│   └── test_analysis.py
├── README.md
└── requirements.txt


----
## Installation.
1. Run pip install -r requirements.txt to install the required packages
2. Run  ../make_dir_structure.sh to create the project structure
3. config.py -> src/config.py
4. downloader.py -> src/data/downloader.py
5. python -m src.data.downloader to download 1m of major Crypto crosses vs Tether (USDT) from binance at 1min intervals.
6. The notebooks folder contains basic EDA for the data
7. The analysis folder contains the granger causality analysis
8. The visualization folder has various visualizations of the data
9. The examples folders shows how to use the code



