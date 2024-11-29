Our first day was pretty heavy and we covered a lot on the board in class. I also realized there were a lot of things they needed to learn and 
starting out with them on day 1 would leave the rest for projects.

We covered or touched on:
1. SABR model and Fixed Income Vol Surfaces. Where to find them in Bloomberg
2. How Exotics structuring works and what the structuring desk does
3. How QIS was born from Structuring and what a QIS teams does -- excess returns swaps, delta-1 trading and 'synthetic asset mgt'
4. Trend following, momentum 
5. Using MACD for trend (and MR), both using ewm() and rolling() in pandas.
6. 200 years of Trend Following by Bouchaud et al
7. Baz et al from AHL on Trend, Carry, and Value and forming portflios, including the AHL signals
8. Mean Reversion strategies, what they are and how they work
9. US Treasury Bond auctions, ONTRs and OFTRs, Stripping and reconstitution, WI Trading
10. Swaps and CMS (a tiny bit), mentioning TEC-10 and the French penchant for CMS10Y puts

I gave them a Collab  notebook and asked them to fix it.
The notebook took 1min  EURUSD bars (freely available for 1year), plotted them, had a Feature ABC and a FeatureManager class for coordinating feature calculations. It included MACD features, Garman-Klass vol, and an estimate of Slippage from HLOC bars. I also did a weekly refit, OLS model. The effective SR using the correlation formula was shown (and I went over this) and SR was around 2.2.  
I then gave them a library, called EWRLS and Recursive Least Squares and asked them to choose how to use them.

The codebase was a bit old and required some alterations to get it to work. Some got the RLS working right away, some struggled on EWRLS. I told them I would give them updated ones the next day. All good and the following day and the supplied EWRLS worked out of the box, as did the RLS.
They spent time producing forecasts etc, and trying to hit various targets.

I then told them about Performance Measurement: Sharpe, Sortino, Calmar, etc. I referenced "How Sharp is the Sharpe Ratio" by StatsPro which has all relevant examples in it. I gave them a performance.py class that crcreated tables out of all the various performance metrics.

Other topics we covered, depending on the questions were:
1. Regularisation and ridge regressions.
2. Rank-deficiency 
3. ARMA models, model selection, AIC, BIC, etc.
4. Test/Train validation splits
5. The importance of the 'lookahead' bias and how to avoid it
6. Basics of MVO and including transaction costs (trading at "top of book"). How to solve it in closed form for a single asset and how you can turn forecasted returns, vol, bid-ask spread and current position to get an optimal trade size. 
7. We spoke a little about Ed Thorpe and Kelly betting.

Finally, I gave projects
1 Most people worked on the data set I gave or downloaded themselves. CL contracts (1-16 are liquid). They were meant to interpolate to 1m, 2m, 3m constant maturity points, do PCA on them (in Levels, not diffs), extract 3 factors and residuals, investigate the residuals and the mean-reversion properties of them, then form portfolios which took advantage of this (e.g., factor1 and factor2 neutral butterflies).

A few asked for special projects. Oscar, Prathnam and Jiang Wei worked on hedging FX vol using a Heston-like model. I helped them to find the data in BBG, fit curves, update etc. Then worked on various delta-hedging methods. This was related to projects they had using SABR for the vol desk (but BBG doesn't have good swaption skew data). 

Ben was psyched with forecasting and the AHL paper so asked about using his 5 min BTC data to forecast a long/flat strategy. He implmented the AHL paper exactly with more MACDs. we then talked about using EWRLS or some other dynamic model (or how to explore it using OLS at different lookbacks). Still a WIP, but he's quite excited to get on with it. 
