# Assuming you have:
# strategy_rets - pd.Series of 1-minute returns
# wts - pd.DataFrame of portfolio weights
# pnl - pd.Series of cumulative PnL (strategy_rets.cumsum())

# Generate complete analysis
analysis = generate_complete_analysis(strategy_rets, wts)

# Print each component
print("Performance Metrics:")
print(analysis['Performance Metrics'])
print("\nReturn Distribution:")
print(analysis['Return Distribution'])
print("\nTrading Activity:")
print(analysis['Trading Activity'])

# If you want to analyze specific time periods:
# For example, analyzing 2023 performance
year_2023_rets = strategy_rets['2023']
year_2023_wts = wts['2023']
analysis_2023 = generate_complete_analysis(year_2023_rets, year_2023_wts)

# Compare different periods:
periods = {
    'Full Sample': strategy_rets,
    '2023': strategy_rets['2023'],
    'Last 3 Months':
    strategy_rets[-3 * 21 * 390:],  # Approximately last 3 months
}

period_analysis = {}
for period_name, rets in periods.items():
    period_analysis[period_name] = analyze_strategy_performance(rets)

# Convert to a single DataFrame for comparison
period_comparison = pd.concat(period_analysis, axis=1)
print("\nPeriod Comparison:")
print(period_comparison)
