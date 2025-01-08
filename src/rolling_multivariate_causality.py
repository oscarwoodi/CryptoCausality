import os
import sys
import sqlite3
import logging
import concurrent.futures
import pandas as pd
from statsmodels.tsa.api import VAR

sys.path.append(os.path.dirname(os.path.abspath('')))
from src.utils.load_data import load_parquet_data

def save_to_db(result_dict, db_file="data/results.db"):
    conn = sqlite3.connect(db_file)
    for token, data in result_dict.items():
        if token == 'lag_order':
            data.to_sql(f"{token}", conn, if_exists='replace')
        else:
            data['stats'].to_sql(f"{token}_stats", conn, if_exists='replace')
            data['preds'].to_sql(f"{token}_preds", conn, if_exists='replace')
    conn.close()

def load_from_db(db_file="data/results.db"):
    conn = sqlite3.connect(db_file)
    result_dict = {}
    lag_order = pd.read_sql("SELECT * FROM lag_order", conn, index_col='index')
    result_dict['lag_order'] = lag_order
    for token in lag_order.columns:
        stats = pd.read_sql(f"SELECT * FROM {token}_stats", conn, index_col='index')
        preds = pd.read_sql(f"SELECT * FROM {token}_preds", conn, index_col='index')
        result_dict[token] = {'stats': stats, 'preds': preds}
    conn.close()
    return result_dict

def rolling_multivariate_causality_v2(
    returns_data,
    window_size: int,
    max_lags: int,
    sig_level: float = 0.05,
    checkpoint_interval: int = 100,
    db_file: str = "data/results.db",
    fit_freq: int = 1
) -> dict:
    """
    Run rolling multivariate Granger causality analysis using VAR model for all tokens!

    Args:
        returns_data: DataFrame of all cryptocurrency returns
        window_size: Size of rolling window
        max_lag: Maximum lag order for VAR model (default: None, uses AIC)
        sig_level: Significance level for p-values (default: 0.05)
        checkpoint_interval: Interval at which to save checkpoints
        db_file: SQLite database file to save results

    Returns:
        Dictionary containing:
        - Test statistics for each variable
        - Coefficient p-values for each variable
        - Optimal lag order
    """
    # Set up logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Running multivariate causality analysis for all tokens")

    # Prepare data
    data = returns_data.dropna()

    # Ensure the data index is a datetime index
    data.index = pd.to_datetime(data.index)

    # Initialize result data structures
    stat_results = pd.DataFrame(columns=["f_stat", "p_value", "significant"], index=data.index)
    predictions = pd.DataFrame(columns=["pred", "significant_tokens"], index=data.index)
    lag_order = pd.DataFrame(columns=["lag_order"], index=data.index)
    result_dict = {token: {'stats': stat_results.copy(), 'preds': predictions.copy()} for token in data.columns}
    result_dict['lag_order'] = lag_order

    # Load checkpoint if available
    if os.path.exists(db_file):
        result_dict = load_from_db(db_file)
        start_index = data.index.get_loc(result_dict['lag_order'].dropna().index[-1]) + 1
    else:
        start_index = 0

    def process_window(start):
        logger.info(
            f"Fitting model for timestep {start}/{len(data) - window_size + 1}..."
        )

        date = data.index[start + window_size - 1]
        window_data = data.iloc[start : start + window_size]

        # Fit VAR model
        model = VAR(window_data)
        results = model.fit(maxlags=max_lags, ic="aic")  # this automatically selects the best lag order

        # Predict next timestep
        lag_order = results.k_ar
        
        if lag_order == 0:
            for i in range(0, fit_freq):
                logger.info(
                    f"No relevant lags for timestep {start+i}/{len(data) - window_size + 1}..."
                        )
                # get new data window
                date = data.index[start + window_size - 1 + i]
                # save lag order
                result_dict['lag_order'].loc[date] = lag_order

            return
        else: 
            for i in range(0, fit_freq):
                logger.info(
                    f"Getting Prediction for timestep {start+i}/{len(data) - window_size + 1}..."
                        )

                # get new data window
                date = data.index[start + window_size - 1 + i]
                window_data = data.iloc[start + i: start + window_size + i]
                prediction = results.forecast(window_data.values[-lag_order:], 1)

                # save lag order
                result_dict['lag_order'].loc[date] = lag_order

                # save results for each token
                for idx, target in enumerate(data.columns):
                    other_tokens = [token for token in data.columns if token != target]

                    # get predictions 
                    result_dict[target]['preds'].loc[date, "pred"] = prediction[0][idx]  # select result only for target variable

                    # find significant tokens
                    causal_tokens = []
                    for cause in other_tokens: 
                        # Use F-test or Chi-square test statistic
                        if results.test_causality(target, [cause], kind="f").test_statistic < sig_level:
                            causal_tokens.append(cause)
                    result_dict[target]['preds'].loc[date, "significant_tokens"] = causal_tokens

                    # get test statistics and p-values for target variable overall
                    f_stat = results.test_causality(target, other_tokens, kind="f").test_statistic
                    p_value = results.test_causality(target, other_tokens, kind="f").pvalue
                    result_dict[target]['stats'].loc[date] = [f_stat, p_value, p_value < 0.05]

        # Save checkpoint at regular intervals
        if start % checkpoint_interval == 0:
            save_to_db(result_dict, db_file)
            logger.info(f"Checkpoint saved at timestep {start}")

    # Use ThreadPoolExecutor to parallelize the rolling window analysis
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_window, range(start_index, len(data) - window_size + 1, fit_freq))

    # Save final results
    save_to_db(result_dict, db_file)
    logger.info("Final results saved")

    return result_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rolling multivariate Granger causality analysis.")
    parser.add_argument("--data_file", type=str, default="../data/processed/", help="Path to the input data file (CSV format).")
    parser.add_argument("--window_size", type=int, default=300, help="Size of the rolling window.")
    parser.add_argument("--max_lags", type=int, default=10, help="Maximum number of lags for the VAR model.")
    parser.add_argument("--sig_level", type=float, default=0.05, help="Significance level for p-values.")
    parser.add_argument("--checkpoint_interval", type=int, default=250, help="Interval at which to save checkpoints.")
    parser.add_argument("--db_file", type=str, default="../data/results.db", help="SQLite database file to save results.")
    parser.add_argument("--interval", type=str, default="1m", help="Interval for the data.")
    parser.add_argument("--fit_freq", type=int, default=1, help="Interval for the data.")

    args = parser.parse_args()

    # Example usage:
    # python rolling_multivariate_causality.py data/processed/ --data_file "../data/processed/ --window_size 300 --max_lags 30 --sig_level 0.05 --checkpoint_interval 100 --db_file data/results.db --interval 1m  --fit_freq 1

    returns, prices = load_parquet_data(data_dir=args.data_file, interval=args.interval)
    log_returns = pd.DataFrame({key: returns[key].set_index('timestamp')["log_returns"] for key in returns.keys()}).dropna()

    # Run analysis
    result_dict = rolling_multivariate_causality_v2(
        log_returns[:1000],
        window_size=args.window_size,
        max_lags=args.max_lags,
        sig_level=args.sig_level,
        checkpoint_interval=args.checkpoint_interval,
        db_file=args.db_file,
        fit_freq=args.fit_freq
    )

    # print final results
    print(result_dict)
