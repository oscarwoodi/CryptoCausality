
# Import required libraries
import pandas as pd
import numpy as np
import zstandard as zstd
import json
from pathlib import Path
import glob
from datetime import datetime

class DatabentoLoader:
    """
    Class to load and process Databento oil futures data
    """
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.symbol_map = None
        self.loaded_data = {}

    def load_symbol_mappings(self):
        """Load and process symbol mapping information"""
        # Read the symbology CSV file
        symbol_df = pd.read_csv(self.base_path / 'mini_symbology.csv')
        self.symbol_map = symbol_df.groupby(
						'raw_symbol')['instrument_id'].first().to_dict()
        return self.symbol_map

    def decompress_zst(self, file_path):
        """Decompress a .zst file and return its contents"""
        with open(file_path, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.stream_reader(fh)
            return pd.read_csv(decompressed)

    def load_ohlcv_files(self, pattern="glbx-mdp3-*.ohlcv-1m.*.csv.zst"):
        """Load all OHLCV files matching the pattern"""
        files = glob.glob(str(self.base_path / pattern))

        for file in files:
            try:
                # Extract symbol from filename
                symbol = file.split('ohlcv-1m.')[-1].split('.csv.zst')[0]

                # Load and process the data
                df = self.decompress_zst(file)

                # Assume standard OHLCV format with timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                df.set_index('timestamp', inplace=True)

                # Store in dictionary
                self.loaded_data[symbol] = df

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    def create_consolidated_df(self):
        """
        Create a consolidated multi-level DataFrame
			with symbols and OHLCV columns
        """
        # Initialize empty DataFrames for each OHLCV component
        consolidated = {}

        for symbol, df in self.loaded_data.items():
            # For each OHLCV component
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in consolidated:
                    consolidated[col] = pd.DataFrame()
                consolidated[col][symbol] = df[col]

        # Create multi-level DataFrame
        final_df = pd.concat(
            [consolidated[col] for col in ['open',
											'high',
											'low',
											'close',
											'volume']],
            keys=['open', 'high', 'low', 'close', 'volume'],
            axis=1
        )

        return final_df

# Example usage
def load_and_prepare_data(base_path):
    """Helper function to load and prepare the data"""
    loader = DabentoLoader(base_path)

    # Load symbol mappings
    symbol_map = loader.load_symbol_mappings()
    print(f"Loaded {len(symbol_map)} symbol mappings")

    # Load OHLCV data
    loader.load_ohlcv_files()
    print(f"Loaded data for {len(loader.loaded_data)} symbols")

    # Create consolidated DataFrame
    final_df = loader.create_consolidated_df()
    print(f"Created consolidated DataFrame with shape {final_df.shape}")

    return final_df, loader
