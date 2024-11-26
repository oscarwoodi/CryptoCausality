# Example usage
# Assuming df is your DataFrame with forward_return and features

# Run both models with default parameters
forecasts = run_both_models(df)

# Or run with custom parameters
forecasts = run_both_models(
    df,
    rolling_window=126,  # 6 months
    ewrls_span=63,  # ~3 months
    regularization=0.5)

# Look at forecast statistics
print("Forecast Statistics:")
print(forecasts.describe())

# Calculate correlation with actual returns
correlations = forecasts.corrwith(df['forward_return'])
print("\nCorrelations with actual returns:")
print(correlations)
