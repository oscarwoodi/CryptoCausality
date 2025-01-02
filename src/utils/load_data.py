# imports
import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import logging

def load_parquet_data(data_dir: str = None, interval: str = "1m", logger=None) -> pd.DataFrame:
    """Initialize analyzer with data directory path."""

    if logger is None:
        # Set up logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)

    if data_dir is None:
        # Get the path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data", "processed")
    else:
        data_dir = data_dir

    logger.info(f"Initializing analyzer with data directory: {data_dir}")

    logger.info("Starting data loading process...")
    all_returns = {}
    all_data = {}

    # Get absolute path for better debugging
    abs_data_dir = os.path.abspath(data_dir)
    logger.info(f"Looking for parquet files in: {abs_data_dir}")

    # Check if directory exists
    if not os.path.exists(abs_data_dir):
        logger.error(f"Directory does not exist: {abs_data_dir}")
        return pd.DataFrame()

    parquet_files = glob.glob(os.path.join(abs_data_dir, "*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")
    if len(parquet_files) == 0:
        logger.info("Contents of directory:")
        for item in os.listdir(abs_data_dir):
            logger.info(f"  {item}")

    for file in parquet_files:
        symbol = os.path.basename(file).split("_")[0]
        time = os.path.basename(file).split("_")[1]
        logger.info(f"Processing file for {symbol}")

        if time == interval:
            try:
                df = pq.read_table(file).to_pandas()
                logger.info(f"Loaded {len(df)} rows for {symbol}")

                if "simple_returns" not in df.columns:
                    # Calculate simple returns
                    logger.info(f"Calculating simple returns for {symbol}")
                    df["simple_returns"] = df["close"].pct_change()

                if "price_diff" not in df.columns:
                    # Calculate absolute difference
                    logger.info(f"Calculating price difference for {symbol}")
                    df["price_diff"] = df["close"].diff()

                if "log_returns" not in df.columns:
                    logger.info(f"Calculating log returns for {symbol}")
                    df["log_returns"] = np.log(df["close"]).diff()  # find log returns

                all_data[symbol] = df
                all_returns[symbol] = df[
                    ["timestamp", "log_returns", "simple_returns", "price_diff"]
                ].copy()

                logger.info(f"Successfully processed {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

    if not all_returns:
        logger.error("No data was loaded!")
        return pd.DataFrame()
    
    # save data to df
    returns_df = pd.DataFrame(all_returns).dropna()
    data_df = pd.DataFrame(all_data).dropna()

    logger.info(f"Final DataFrame shape: {returns_df.shape}")
    logger.info(f"Columns: {returns_df.columns.tolist()}")
    return returns_df, data_df