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
from src.visualization.causality_viz import CausalityVisualizer
import logging

# ... (previous code remains the same until main function) ...

def main():
    """Main analysis pipeline."""
    try:
        # Initialize analyzer
        analyzer = AutomatedGrangerAnalyzer()

        # Run analysis
        logger.info("Running automated Granger causality analysis...")
        results = analyzer.analyze_all_pairs()

        if results is None or len(results) == 0:
            print("\nNo Granger causality results were generated.")
            return

        # Print all results first for debugging
        print("\nAll Granger Causality Tests:")
        print(results)

        # Safely filter significant results
        if 'significant' in results.columns:
            significant_mask = results['significant'] == True
            significant_results = results[significant_mask]
            
            if len(significant_results) > 0:
                print("\nSignificant Granger Causality Relationships:")
                for _, row in significant_results.iterrows():
                    print(f"{row['cause']} -> {row['effect']}")
                    print(f"  Optimal lag: {row['optimal_lag']}")
                    print(f"  F-statistic: {row['f_stat']:.4f}")
                    print(f"  P-value: {row['p_value']:.4f}\n")
            else:
                print("\nNo significant Granger causality relationships found.")
        else:
            print("\nError: 'significant' column not found in results.")
            print("Available columns:", results.columns.tolist())

    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()