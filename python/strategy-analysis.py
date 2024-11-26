import pandas as pd
import numpy as np
from performance import (
    annual_return, annual_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, omega_ratio, stability_of_timeseries,
    tail_ratio, common_sense_ratio, max_drawdown_length, value_at_risk
)

def analyze_strategy_performance(strategy_rets, trading_minutes=390, trading_days=252):
    """
    Comprehensive performance analysis for a trading strategy
    
    Parameters:
    -----------
    strategy_rets : pd.Series
        Series of strategy returns at 1-minute frequency
    trading_minutes : int
        Number of trading minutes per day (default 390 for US equity market)
    trading_days : int
        Number of trading days per year
        
    Returns:
    --------
    pd.DataFrame with performance metrics
    """
    # Calculate total trading periods per year
    trading_periods = trading_minutes * trading_days
    
    # Group returns by day for daily statistics
    daily_rets = strategy_rets.groupby(pd.Grouper(freq='D')).sum()
    
    # Calculate key metrics
    metrics = {
        'Total Return (%)': strategy_rets.sum() * 100,
        'Annualized Return (%)': annual_return(strategy_rets, trading_days=trading_periods) * 100,
        'Annualized Volatility (%)': annual_volatility(strategy_rets, trading_days=trading_periods) * 100,
        'Sharpe Ratio': sharpe_ratio(strategy_rets, trading_days=trading_periods),
        'Sortino Ratio': sortino_ratio(strategy_rets, trading_days=trading_periods),
        'Max Drawdown (%)': max_drawdown(strategy_rets) * 100,
        'Calmar Ratio': calmar_ratio(strategy_rets, trading_days=trading_periods),
        'Omega Ratio': omega_ratio(strategy_rets, trading_days=trading_periods),
        'Stability': stability_of_timeseries(strategy_rets),
        'Tail Ratio': tail_ratio(strategy_rets),
        'Common Sense Ratio': common_sense_ratio(strategy_rets),
        'Daily VaR 99% (%)': value_at_risk(daily_rets, horizon=1, pctile=0.99) * 100,
        'Daily Hit Rate (%)': (daily_rets > 0).mean() * 100,
        'Daily Win/Loss Ratio': abs(daily_rets[daily_rets > 0].mean() / daily_rets[daily_rets < 0].mean()),
    }
    
    # Add drawdown duration metrics
    dd_lengths = max_drawdown_length(strategy_rets)
    metrics.update({
        'Max DD Peak to Trough (mins)': dd_lengths['peak_to_trough_maxdd'],
        'Max DD Peak to Peak (mins)': dd_lengths['peak_to_peak_maxdd'],
        'Longest DD Period (mins)': dd_lengths['peak_to_peak_longest']
    })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df = metrics_df.set_index('Metric')
    
    return metrics_df

def analyze_returns_distribution(strategy_rets):
    """
    Analyze the distribution of returns
    
    Parameters:
    -----------
    strategy_rets : pd.Series
        Series of strategy returns
        
    Returns:
    --------
    pd.DataFrame with distribution metrics
    """
    metrics = {
        'Mean (bps)': strategy_rets.mean() * 10000,
        'Std Dev (bps)': strategy_rets.std() * 10000,
        'Skewness': strategy_rets.skew(),
        'Kurtosis': strategy_rets.kurtosis(),
        'Minimum (bps)': strategy_rets.min() * 10000,
        '1st Percentile (bps)': strategy_rets.quantile(0.01) * 10000,
        '5th Percentile (bps)': strategy_rets.quantile(0.05) * 10000,
        'Median (bps)': strategy_rets.median() * 10000,
        '95th Percentile (bps)': strategy_rets.quantile(0.95) * 10000,
        '99th Percentile (bps)': strategy_rets.quantile(0.99) * 10000,
        'Maximum (bps)': strategy_rets.max() * 10000,
        'Positive Returns (%)': (strategy_rets > 0).mean() * 100,
        'Zero Returns (%)': (strategy_rets == 0).mean() * 100,
        'Negative Returns (%)': (strategy_rets < 0).mean() * 100
    }
    
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df = metrics_df.set_index('Metric')
    
    return metrics_df

def analyze_trading_activity(strategy_rets, wts):
    """
    Analyze trading activity and portfolio turnover
    
    Parameters:
    -----------
    strategy_rets : pd.Series
        Series of strategy returns
    wts : pd.DataFrame
        DataFrame of portfolio weights
        
    Returns:
    --------
    pd.DataFrame with activity metrics
    """
    # Calculate weight changes
    weight_changes = wts.diff().abs().sum(axis=1)
    
    metrics = {
        'Average Daily Turnover (%)': weight_changes.mean() * 100,
        'Median Daily Turnover (%)': weight_changes.median() * 100,
        'Maximum Daily Turnover (%)': weight_changes.max() * 100,
        'Average Position Count': (wts != 0).sum(axis=1).mean(),
        'Average Long Position Count': (wts > 0).sum(axis=1).mean(),
        'Average Short Position Count': (wts < 0).sum(axis=1).mean(),
        'Average Gross Exposure': wts.abs().sum(axis=1).mean(),
        'Average Net Exposure': wts.sum(axis=1).mean(),
        'Average Long Exposure (%)': (wts[wts > 0].sum(axis=1)).mean() * 100,
        'Average Short Exposure (%)': (wts[wts < 0].sum(axis=1)).mean() * 100
    }
    
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df = metrics_df.set_index('Metric')
    
    return metrics_df

def generate_complete_analysis(strategy_rets, wts, trading_minutes=390, trading_days=252):
    """
    Generate a complete analysis report
    
    Parameters:
    -----------
    strategy_rets : pd.Series
        Series of strategy returns
    wts : pd.DataFrame
        DataFrame of portfolio weights
    trading_minutes : int
        Number of trading minutes per day
    trading_days : int
        Number of trading days per year
        
    Returns:
    --------
    dict of DataFrames with different analysis components
    """
    return {
        'Performance Metrics': analyze_strategy_performance(strategy_rets, trading_minutes, trading_days),
        'Return Distribution': analyze_returns_distribution(strategy_rets),
        'Trading Activity': analyze_trading_activity(strategy_rets, wts)
    }
