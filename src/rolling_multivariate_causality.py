import os
import sys
import pickle
import logging
import concurrent.futures
import pandas as pd
from statsmodels.tsa.api import VAR

sys.path.append(os.path.dirname(os.path.abspath('')))
from src.utils.load_data import load_parquet_data


def save_checkpoint(result_dict, filename="data/checkpoints/rmc_checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(result_dict, f)

def load_checkpoint(filename="data/checkpoints/rmc_checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def rolling_multivariate_causality_v2(
    returns_data,
    window_size: int,
    max_lags: int,
    sig_level: float = 0.05,
    checkpoint_interval: int = 100,
    checkpoint_file: str = "data/checkpoints/rmc_checkpoint.pkl"
) -> dict:
    """
    Run rolling multivariate Granger causality analysis using VAR model for all tokens!

    Args:
        returns_data: DataFrame of all cryptocurrency returns
        window_size: Size of rolling window
        max_lag: Maximum lag order for VAR model (default: None, uses AIC)
        sig_level: Significance level for p-values (default: 0.05)
        checkpoint_interval: Interval at which to save checkpoints
        checkpoint_file: File to save checkpoints

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
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        result_dict = checkpoint
        start_index = max(result_dict['lag_order'].dropna().index) + 1
    else:
        start_index = 0

    def process_window(start):
        logger.info(
            f"Running multivariate causality analysis for timestep {start}/{len(data) - window_size + 1}..."
        )

        date = data.index[start + window_size - 1]
        window_data = data.iloc[start : start + window_size]

        # Fit VAR model
        model = VAR(window_data)
        results = model.fit(maxlags=max_lags, ic="aic")  # this automatically selects the best lag order

        # Predict next timestep
        lag_order = results.k_ar
        result_dict['lag_order'].loc[date] = lag_order
        
        if lag_order == 0:
            return
        else: 
            prediction = results.forecast(window_data.values[-lag_order:], 1)

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
            save_checkpoint(result_dict, checkpoint_file)
            logger.info(f"Checkpoint saved at timestep {start}")

    # Use ThreadPoolExecutor to parallelize the rolling window analysis
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_window, range(start_index, len(data) - window_size + 1))

    # Save final results
    save_checkpoint(result_dict, checkpoint_file)
    logger.info("Final results saved")

    return result_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rolling multivariate Granger causality analysis.")
    parser.add_argument("--data_file", type=str, default="../data/processed/", help="Path to the input data file (CSV format).")
    parser.add_argument("--window_size", type=int, default=300, help="Size of the rolling window.")
    parser.add_argument("--max_lags", type=int, default=30, help="Maximum number of lags for the VAR model.")
    parser.add_argument("--sig_level", type=float, default=0.05, help="Significance level for p-values.")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval at which to save checkpoints.")
    parser.add_argument("--checkpoint_file", type=str, default="data/checkpoints/rmc_checkpoint.pkl", help="File to save checkpoints.")
    parser.add_argument("--interval", type=str, default="1m", help="Interval for the data.")

    args = parser.parse_args()

    # Example usage:
    # python rolling_multivariate_causality.py data/processed/ --window_size 300 --max_lags 30 --sig_level 0.05 --checkpoint_interval 100 --checkpoint_file checkpoint.pkl --interval 1m

    returns, prices = load_parquet_data(data_dir=args.data_file, interval=args.interval)
    log_returns = pd.DataFrame({key: returns[key].set_index('timestamp')["log_returns"] for key in returns.keys()}).dropna()

    # Ensure the data index is a datetime index
    log_returns.index = pd.to_datetime(log_returns.index)

    # Run analysis
    result_dict = rolling_multivariate_causality_v2(
        log_returns[-302:],
        window_size=args.window_size,
        max_lags=args.max_lags,
        sig_level=args.sig_level,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_file=args.checkpoint_file
    )

    # print final results
    print(result_dict)

    # Save final results
    with open("data/checkpoints/rmv_final_results.pkl", "wb") as f:
        pickle.dump(result_dict, f)
