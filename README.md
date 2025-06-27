# Overview
An algorithmic trading strategy for SOXL(3x leveraged semiconductor ETF that tracks the ICE Semiconductor Sector Index) that outperforms base SOXL (buy-and-hold) returns. Although SOXL is highly volatile, and, thus, have a striking -90%+ maximum drawdown, my algorithm makes timely trades to leverage its extremely high volatility to generate high alpha. 

# Backtesting results
a) Since inception (from 2010-03-11 to 2025-06-20):
  - Total Return: 25992.90%
  - Buy-and-Hold Return: 3171.8%
  - Number of Trades: 46
  - Win Rate: 92.86%
  - Strategy Maximum Drawdown: -61.79%
  - Buy-and-Hold Maximum Drawdown: -90.51%
  - Alpha (Annualized): 30.02%
  - Beta: 0.49
  - CAGR: 43.94%
  - Expected Return (Annualized): 65.61%
  - Sharpe Ratio: 1.17
  - Strategy Volatility (Annualized): 52.48%
  - Buy-and-Hold Volatility (Annualized): 88.60%


b) 15 years (from 2010-06-20 to 2025-06-20): 
  - Total Return: 29671.44%
  - Buy-and-Hold Return: 3234%
  - Number of Trades: 47
  - Win Rate: 92.86%
  - Strategy Maximum Drawdown: -61.79%
  - Buy-and-Hold Maximum Drawdown: -90.51%
  - Alpha (Annualized): 30.12%
  - Beta: 0.51
  - CAGR: 46.19%
  - Sharpe Ratio: 1.22
  - Strategy Volatility (Annualized): 53.31%
  - Buy-and-Hold Volatility (Annualized): 88.03%

c) 10 years (from 2015-06-20 to 2025-06-20): 
  - Total Return: 8372.89%
  - Buy-and-Hold Return: 742.4%
  - Number of Trades: 34
  - Win Rate: 90.48%
  - Strategy Maximum Drawdown: -61.79%
  - Buy-and-Hold Maximum Drawdown: -90.51%
  - Alpha (Annualized): 40.13%
  - Beta: 0.48
  - CAGR: 55.88%
  - Sharpe Ratio: 1.41
  - Strategy Volatility (Annualized): 57.29%
  - Buy-and-Hold Volatility (Annualized): 96.66%

d) 5 years (from 2020-01-01 to 2025-06-20):
  - Total Return: 1061.69%
  - Buy-and-Hold Return: 11.0%
  - Number of Trades: 17
  - Win Rate: 83.33%
  - Strategy Maximum Drawdown: -55.69%
  - Buy-and-Hold Maximum Drawdown: -90.51%
  - Alpha (Annualized): 52.33%
  - Beta: 0.42
  - CAGR: 56.60%
  - Sharpe Ratio: 1.41
  - Strategy Volatility (Annualized): 61.28%
  - Buy-and-Hold Volatility (Annualized): 113.22%


# Uses of Machine Learning and Optimization in the algorithm
a) Hyperparameter Optimization using scikit-optimize (Bayesian Optimization)
b) Grid Search
c) Scipy Optimization for continuous parameter fine-tuning
d) Monte Carlo Simulations

# Challenges I ran into: 
a) SOXL is a relatively young financial product as it was listed on NYSE just a little over 15 years ago. Thus, we were not able to backtest SOXL's performance during the extremely volatile 2002 Dot Com Bubble Bust and 2008 Financial Crisis. 
b) There were minor discrepancies with RSI and Bollinger Bands data from Python's yfinance library and the online platform TradingView, but I adhered to yfinance data.

# Get Started
  1. Install necessary libraries/packages
     pip install yfinance pandas numpy matplotlib scipy scikit-optimize ta
  2. Modify the start_date and end_date to test the strategy's performance for various periods.















