# examples/automated_gc.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import glob
from typing import Dict, Tuple, List
import logging

# local imports
from src.data.data_processor import DataProcessor
from src.visualization.causality_viz import CausalityVisualizer
from src.analysis.causality import CausalityAnalyzer
from src.analysis.granger_causality import AutomatedGrangerAnalyzer
from src.analysis.stationarity import StationarityTester
from src.analysis.outliers import OutlierAnalyzer
from src.utils.helpers import calculate_returns

def load_data(logger, data_dir: str = None) -> pd.DataFrame:
    """Initialize analyzer with data directory path."""
    if data_dir is None:
        # Get the path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(script_dir), "data", "processed")
    else:
        data_dir = data_dir
    logger.info(f"Initializing analyzer with data directory: {data_dir}")

    logger.info("Starting data loading process...")
    all_returns = {}
    
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
        symbol = os.path.basename(file).split('_')[0]
        logger.info(f"Processing file for {symbol}")
        
        try:
            df = pq.read_table(file).to_pandas()
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
            if 'log_returns' not in df.columns:
                logger.info(f"Calculating log returns for {symbol}")
                df['log_returns'] = np.log(df['close']).diff()
            
            all_returns[symbol] = df['log_returns']
            logger.info(f"Successfully processed {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not all_returns:
        logger.error("No data was loaded!")
        return pd.DataFrame()
        
    returns_df = pd.DataFrame(all_returns).dropna()
    logger.info(f"Final DataFrame shape: {returns_df.shape}")
    logger.info(f"Columns: {returns_df.columns.tolist()}")
    return returns_df

def run_stationarity_analysis(data: pd.DataFrame, logger) -> None:
    """Run and print stationarity analysis."""
    logger.info("Running stationarity analysis...")
    
    tester = StationarityTester(data)
    summary = tester.get_stationarity_summary()
    
    print("\nStationarity Analysis Results:")
    print(summary)
    
    return summary

def run_outlier_analysis(data: pd.DataFrame, logger) -> None:
    """Run and print outlier analysis."""
    logger.info("Running outlier analysis...")
    
    analyzer = OutlierAnalyzer(data)
    summary = analyzer.get_summary_statistics()
    
    print("\nOutlier Analysis Results:")
    print(summary)
    
    return summary

def run_causality_analysis(data: pd.DataFrame, logger) -> None:
    """Run and print causality analysis."""
    logger.info("Running causality analysis...")
    
    analyzer = CausalityAnalyzer(data)
    results = analyzer.analyze_all_causality()
    metrics = analyzer.get_causality_metrics()
    
    print("\nCausality Analysis Results:")
    print("\nGranger Causality Summary:")
    print(results['granger'])
    
    print("\nCorrelation Structure:")
    print(results['correlation'])
    
    print("\nCausality Metrics:")
    print(metrics)
    
    return results, metrics

def run_granger_causality_analysis(data: pd.DataFrame, logger) -> None:
    # Granger Analysis
    try:
        # Initialize analyzer
        analyzer = AutomatedGrangerAnalyzer(data)
        
        # Run analysis
        logger.info("Running pairwise causality tests...")
        results = analyzer.analyze_all_pairs()
        
        if results is None or len(results) == 0:
            logger.warning("No Granger causality results were generated.")
            return

        # Print all results
        print("\nAll Granger Causality Tests:")
        print(results.to_string())
        
        # Print significant results
        if 'significant' in results.columns:
            significant_results = results[results['significant'] == True]
            
            if len(significant_results) > 0:
                print("\nSignificant Granger Causality Relationships:")
                for _, row in significant_results.iterrows():
                    print(f"\n{row['cause']} -> {row['effect']}")
                    print(f"  Optimal lag: {row['optimal_lag']}")
                    print(f"  F-statistic: {row['f_stat']:.4f}")
                    print(f"  P-value: {row['p_value']:.4f}")
                
            else:
                print("\nNo significant Granger causality relationships found.")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}", exc_info=True)
        raise

    return results

    
def main():
    """Main analysis pipeline."""
    # Set up logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Granger causality analysis")
    
     # Load data
    data = load_data(logger)

    # Run analyses
    stationarity_results = run_stationarity_analysis(data, logger)
    outlier_results = run_outlier_analysis(data, logger)
    causality_results, causality_metrics = run_causality_analysis(data, logger)
    granger_causality_results = run_granger_causality_analysis(data, logger)

    # save results if needed
    stationarity_results.to_csv("results/stationarity_results.csv")
    outlier_results.to_csv("results/outlier_results.csv")
    causality_results['granger'].to_csv("results/grangerv1_causality_results.csv")
    causality_results['correlation'].to_csv("results/correlation_causality_results.csv")
    causality_results['instantaneous'].to_csv("results/instantaneous_causality_results.csv")
    causality_metrics.to_csv("results/causality_metrics.csv")
    granger_causality_results.to_csv("results/grangerv2_causality_results.csv")


if __name__ == "__main__":
    main()