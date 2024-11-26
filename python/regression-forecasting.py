import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    """
    Prepare features and target for regression
    """
    # Create feature matrix X
    feature_cols = ['mean_reversion_signal1', 'mean_reversion_signal2', 'GKVol', 'bid_ask_spread']
    X = df[feature_cols].values
    
    # Create target variable y
    y = df['forward_return'].values.reshape(-1, 1)
    
    return X, y, feature_cols

def rolling_regression_forecast(df, window_size=252):
    """
    Implement rolling regression using RecursiveOLS
    
    Parameters:
    -----------
    df : pandas DataFrame
        Contains forward_return and features
    window_size : int
        Rolling window size in days
    
    Returns:
    --------
    pandas Series with forecasts
    """
    from recursive_ols import RecursiveOLS
    
    X, y, feature_cols = prepare_data(df)
    forecasts = np.zeros(len(df))
    
    # Initialize with first window
    model = RecursiveOLS(y[:window_size], X[:window_size])
    forecasts[:window_size] = np.nan
    
    # Roll forward making predictions
    for t in range(window_size, len(df)):
        # Update model with new observation
        model.rolling_add(y[t-1:t], X[t-1:t], roll_length=window_size)
        
        # Make forecast using current features
        forecasts[t] = X[t] @ model.beta
        
    return pd.Series(forecasts, index=df.index, name='rolling_forecast')

def ewrls_forecast(df, span=252, regularization=0.1):
    """
    Implement EWRLS regression using EWRLSRidge
    
    Parameters:
    -----------
    df : pandas DataFrame
        Contains forward_return and features
    span : float
        Exponential weighting span (higher = more history weight)
    regularization : float
        L2 regularization parameter
        
    Returns:
    --------
    pandas Series with forecasts
    """
    from ewrls import EWRLSRidge
    
    X, y, feature_cols = prepare_data(df)
    forecasts = np.zeros(len(df))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize EWRLS model
    model = EWRLSRidge(
        num_features=len(feature_cols),
        span=span,
        regularization=regularization,
        history=True
    )
    
    # Initial update with first observation
    model.update(y[0:1], X_scaled[0:1])
    forecasts[0] = np.nan
    
    # Roll forward making predictions
    for t in range(1, len(df)):
        # Make forecast using current features
        forecasts[t] = model.generate_prediction(X_scaled[t])
        
        # Update model with realized return
        if t < len(df) - 1:  # Don't update on last observation
            model.update(y[t:t+1], X_scaled[t:t+1])
    
    return pd.Series(forecasts, index=df.index, name='ewrls_forecast')

def run_both_models(df, rolling_window=252, ewrls_span=252, regularization=0.1):
    """
    Run both rolling regression and EWRLS models
    
    Parameters:
    -----------
    df : pandas DataFrame
        Contains forward_return and features
    rolling_window : int
        Window size for rolling regression
    ewrls_span : float
        Span parameter for EWRLS
    regularization : float
        L2 regularization parameter for EWRLS
        
    Returns:
    --------
    DataFrame with both forecasts
    """
    # Run both models
    rolling_forecasts = rolling_regression_forecast(df, window_size=rolling_window)
    ewrls_forecasts = ewrls_forecast(df, span=ewrls_span, regularization=regularization)
    
    # Combine results
    results = pd.concat([
        rolling_forecasts,
        ewrls_forecasts
    ], axis=1)
    
    return results
