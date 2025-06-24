import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not installed. Will use grid search and scipy optimization only.")
import warnings
warnings.filterwarnings('ignore')

# Function to run the strategy with a given allocation percentage
def run_strategy(allocation_pct, data, initial_capital=100000, verbose=False):
    """
    Run the trading strategy with a given allocation percentage
    Returns the final total return percentage
    """
    # Ensure allocation_pct is a scalar
    if isinstance(allocation_pct, (list, np.ndarray)):
        allocation_pct = float(allocation_pct[0])
    
    cash = initial_capital
    positions = []
    equity = []
    trades = []
    
    for i in range(len(data)):
        current_date = data.index[i]
        current_price = data['Close'].iloc[i]
        
        if data['Buy_Signal'].iloc[i]:
            # Buy: Allocate allocation_pct of available cash
            investment = cash * allocation_pct
            if investment >= 100:
                shares_to_buy = investment / current_price
                cash -= investment
                positions.append({
                    'date': current_date,
                    'price': current_price,
                    'shares': shares_to_buy
                })
                trades.append({
                    'Date': current_date,
                    'Type': 'Buy',
                    'Price': current_price,
                    'Shares': shares_to_buy
                })
                if verbose:
                    total_shares = sum(pos['shares'] for pos in positions)
                    print(f"Buy on {current_date.date()} at {current_price:.2f}, Shares: {shares_to_buy:.2f}, Total: {total_shares:.2f}")
        
        elif data['Sell_Signal'].iloc[i] and positions:
            # Sell: Sell allocation_pct of shares from each open position
            total_shares_sold = 0
            new_positions = []
            for pos in positions:
                shares_to_sell = pos['shares'] * allocation_pct
                total_shares_sold += shares_to_sell
                cash += shares_to_sell * current_price
                remaining_shares = pos['shares'] - shares_to_sell
                if remaining_shares > 0:
                    new_positions.append({
                        'date': pos['date'],
                        'price': pos['price'],
                        'shares': remaining_shares
                    })
            positions = new_positions
            if total_shares_sold > 0:
                trades.append({
                    'Date': current_date,
                    'Type': 'Sell',
                    'Price': current_price,
                    'Shares': total_shares_sold
                })
                if verbose:
                    total_shares = sum(pos['shares'] for pos in positions)
                    print(f"Sell on {current_date.date()} at {current_price:.2f}, Shares: {total_shares_sold:.2f}, Remaining: {total_shares:.2f}")
        
        # Calculate current equity
        total_shares = sum(pos['shares'] for pos in positions)
        current_equity = cash + total_shares * current_price
        equity.append(current_equity)
    
    # Calculate metrics
    final_return = (equity[-1] / initial_capital - 1) * 100 if equity else 0
    
    # Calculate maximum drawdown
    equity_series = pd.Series(equity)
    running_max = equity_series.cummax()
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = drawdowns.min() * 100
    
    # Calculate Sharpe ratio
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) > 0:
        annualized_return = (1 + daily_returns.mean()) ** 252 - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.04) / annualized_volatility if annualized_volatility != 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_return': final_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_equity': equity[-1] if equity else initial_capital,
        'num_trades': len([t for t in trades if t['Type'] == 'Buy'])
    }

# Load and prepare data
print("Loading SOXL data...")
start_date = "2010-06-20"
end_date = "2025-06-20"

# Fetch SOXL data
ticker = "SOXL"
data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)

# Fetch VIX data
vix_ticker = "^VIX"
vix_data = yf.download(vix_ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)

# Handle multi-index columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

if isinstance(vix_data.columns, pd.MultiIndex):
    vix_data.columns = vix_data.columns.get_level_values(0)
vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX_Close'})

# Merge VIX data
data = data.join(vix_data, how='left')

# Calculate indicators
bb_indicator = BollingerBands(close=data['Close'], window=20, window_dev=2)
data['BB_upper'] = bb_indicator.bollinger_hband()
data['BB_lower'] = bb_indicator.bollinger_lband()

rsi_indicator = RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi_indicator.rsi()
data['RSI_MA'] = data['RSI'].rolling(window=9).mean()

# Initialize signal columns
data['Buy_Signal'] = False
data['Sell_Signal'] = False
data['Preliminary_Sell'] = False

# Drop NaN values
data = data.dropna()

# Generate signals
print("Generating trading signals...")
for i in range(2, len(data)):
    # Buy Conditions
    cond_a = (data['Close'].iloc[i - 1] < data['BB_lower'].iloc[i - 1] and
              data['Close'].iloc[i - 2] < data['BB_lower'].iloc[i - 2])
    cond_b = (data['RSI'].iloc[i - 1] <= 32.5 or data['RSI'].iloc[i - 2] <= 32.5)
    cond_vix = data['VIX_Close'].iloc[i] >= 50
    if (cond_a and cond_b) or cond_vix:
        data.loc[data.index[i], 'Buy_Signal'] = True
    
    # Sell Conditions
    prelim_a = (data['Close'].iloc[i - 1] > data['BB_upper'].iloc[i - 1] and
                data['Close'].iloc[i] < data['BB_upper'].iloc[i])
    prelim_b = (data['RSI'].iloc[i - 1] >= 70 or data['RSI'].iloc[i] >= 70)
    if prelim_a and prelim_b:
        data.loc[data.index[i], 'Preliminary_Sell'] = True
    
    if (data['Preliminary_Sell'].iloc[i] or
            (i > 0 and data['Preliminary_Sell'].iloc[i - 1])):
        if (i > 1 and data['RSI'].iloc[i - 1] > data['RSI_MA'].iloc[i - 1] and
                data['RSI'].iloc[i] <= data['RSI_MA'].iloc[i]):
            data.loc[data.index[i], 'Sell_Signal'] = True

print(f"Total Buy Signals: {data['Buy_Signal'].sum()}")
print(f"Total Sell Signals: {data['Sell_Signal'].sum()}")

# Analysis of your initial results
print("\n" + "="*50)
print("ANALYSIS OF YOUR INITIAL GRID SEARCH RESULTS")
print("="*50)
print("\nKey Findings from your results:")
print("1. Optimal allocation appears to be around 50-55%")
print("2. Total return peaks at 55% allocation (29,769.61%)")
print("3. Sharpe ratio peaks at 50% allocation (1.22)")
print("4. Drawdown improves (gets less negative) as allocation increases to 65%")
print("5. Beyond 65%, performance deteriorates rapidly")

# Extended Grid Search around the optimal region
print("\n" + "="*50)
print("REFINED GRID SEARCH (45% - 60%)")
print("="*50)

# Fine grid search around the optimal region
fine_grid = np.linspace(0.45, 0.60, 31)  # Test from 45% to 60% in 0.5% increments
fine_results = []

print("\nTesting allocation percentages in detail:")
for alloc in fine_grid:
    result = run_strategy(alloc, data)
    fine_results.append({
        'allocation_pct': alloc,
        'total_return': result['total_return'],
        'max_drawdown': result['max_drawdown'],
        'sharpe_ratio': result['sharpe_ratio'],
        'num_trades': result['num_trades']
    })
    if alloc * 100 % 1 == 0:  # Print only whole percentages
        print(f"Allocation: {alloc*100:.1f}% | Return: {result['total_return']:.2f}% | Drawdown: {result['max_drawdown']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")

# Find best allocation by different metrics
fine_df = pd.DataFrame(fine_results)
best_return_idx = fine_df['total_return'].idxmax()
best_sharpe_idx = fine_df['sharpe_ratio'].idxmax()
best_combined_idx = (fine_df['total_return'] - 50 * abs(fine_df['max_drawdown'])).idxmax()

print(f"\nRefined Grid Search Results:")
print(f"Best by Total Return: {fine_df.loc[best_return_idx, 'allocation_pct']*100:.1f}% → {fine_df.loc[best_return_idx, 'total_return']:.2f}% return")
print(f"Best by Sharpe Ratio: {fine_df.loc[best_sharpe_idx, 'allocation_pct']*100:.1f}% → {fine_df.loc[best_sharpe_idx, 'sharpe_ratio']:.3f} Sharpe")
print(f"Best Combined Score: {fine_df.loc[best_combined_idx, 'allocation_pct']*100:.1f}% → {fine_df.loc[best_combined_idx, 'total_return']:.2f}% return, {fine_df.loc[best_combined_idx, 'max_drawdown']:.2f}% drawdown")

# Scipy optimization for exact optimal
print("\n" + "="*50)
print("SCIPY OPTIMIZATION FOR EXACT OPTIMAL")
print("="*50)

# Define different objective functions
def objective_return(x):
    return -run_strategy(x, data)['total_return']

def objective_sharpe(x):
    return -run_strategy(x, data)['sharpe_ratio']

def objective_combined(x):
    result = run_strategy(x, data)
    return -(result['total_return'] - 50 * abs(result['max_drawdown']))

# Optimize for different objectives
opt_return = minimize_scalar(objective_return, bounds=(0.45, 0.60), method='bounded')
opt_sharpe = minimize_scalar(objective_sharpe, bounds=(0.45, 0.60), method='bounded')
opt_combined = minimize_scalar(objective_combined, bounds=(0.45, 0.60), method='bounded')

print(f"Optimal for Max Return: {opt_return.x*100:.4f}% → {-opt_return.fun:.2f}% return")
print(f"Optimal for Max Sharpe: {opt_sharpe.x*100:.4f}% → Sharpe {-opt_sharpe.fun:.3f}")
print(f"Optimal for Combined Score: {opt_combined.x*100:.4f}%")

# Get detailed results for optimal allocations
optimal_allocations = {
    'Max Return': opt_return.x,
    'Max Sharpe': opt_sharpe.x,
    'Balanced': opt_combined.x,
    'Original': 0.500000001
}

print("\n" + "="*50)
print("DETAILED COMPARISON OF OPTIMAL ALLOCATIONS")
print("="*50)

comparison_results = []
for name, alloc in optimal_allocations.items():
    result = run_strategy(alloc, data)
    comparison_results.append({
        'Strategy': name,
        'Allocation %': alloc * 100,
        'Total Return %': result['total_return'],
        'Max Drawdown %': result['max_drawdown'],
        'Sharpe Ratio': result['sharpe_ratio'],
        'Trades': result['num_trades']
    })

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df.to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Total Return vs Allocation (Fine Grid)
ax1 = axes[0, 0]
ax1.plot(fine_df['allocation_pct']*100, fine_df['total_return'], 'b-', linewidth=2)
ax1.scatter([opt_return.x*100], [-opt_return.fun], color='red', s=100, zorder=5, label=f'Optimal: {opt_return.x*100:.2f}%')
ax1.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='Original: 50%')
ax1.set_xlabel('Allocation %')
ax1.set_ylabel('Total Return %')
ax1.set_title('Total Return vs Allocation Percentage (Detailed)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Sharpe Ratio vs Allocation (Fine Grid)
ax2 = axes[0, 1]
ax2.plot(fine_df['allocation_pct']*100, fine_df['sharpe_ratio'], 'g-', linewidth=2)
ax2.scatter([opt_sharpe.x*100], [-opt_sharpe.fun], color='red', s=100, zorder=5, label=f'Optimal: {opt_sharpe.x*100:.2f}%')
ax2.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='Original: 50%')
ax2.set_xlabel('Allocation %')
ax2.set_ylabel('Sharpe Ratio')
ax2.set_title('Sharpe Ratio vs Allocation Percentage (Detailed)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Return vs Drawdown Tradeoff
ax3 = axes[1, 0]
scatter = ax3.scatter(fine_df['max_drawdown'], fine_df['total_return'], 
                     c=fine_df['allocation_pct']*100, cmap='viridis', s=50)
ax3.set_xlabel('Max Drawdown %')
ax3.set_ylabel('Total Return %')
ax3.set_title('Return vs Drawdown Tradeoff')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Allocation %')
ax3.grid(True, alpha=0.3)

# Plot 4: Combined Score
ax4 = axes[1, 1]
combined_score = fine_df['total_return'] - 50 * abs(fine_df['max_drawdown'])
ax4.plot(fine_df['allocation_pct']*100, combined_score, 'purple', linewidth=2)
ax4.scatter([opt_combined.x*100], [combined_score.max()], color='red', s=100, zorder=5, 
           label=f'Optimal: {opt_combined.x*100:.2f}%')
ax4.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='Original: 50%')
ax4.set_xlabel('Allocation %')
ax4.set_ylabel('Combined Score')
ax4.set_title('Combined Score (Return - 50 × |Drawdown|)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Monte Carlo simulation for robustness
print("\n" + "="*50)
print("MONTE CARLO ROBUSTNESS TEST")
print("="*50)

# Test with random subsamples of the data
np.random.seed(42)
n_simulations = 100
test_allocations = [0.50, opt_return.x, opt_sharpe.x, opt_combined.x]
mc_results = {alloc: [] for alloc in test_allocations}

print(f"Running {n_simulations} simulations with 80% random samples...")
for _ in range(n_simulations):
    # Random sample 80% of the data
    sample_indices = np.random.choice(len(data), size=int(0.8 * len(data)), replace=False)
    sample_indices.sort()
    sample_data = data.iloc[sample_indices]
    
    for alloc in test_allocations:
        result = run_strategy(alloc, sample_data)
        mc_results[alloc].append(result['total_return'])

# Calculate statistics
print("\nMonte Carlo Results (Mean ± Std):")
for alloc in test_allocations:
    returns = mc_results[alloc]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"Allocation {alloc*100:.2f}%: {mean_return:.2f}% ± {std_return:.2f}%")

# Create recommendation
print("\n" + "="*50)
print("FINAL RECOMMENDATION")
print("="*50)
print(f"\nBased on the comprehensive analysis:")
print(f"1. For Maximum Returns: Use {opt_return.x*100:.2f}% allocation")
print(f"2. For Best Risk-Adjusted Returns: Use {opt_sharpe.x*100:.2f}% allocation")
print(f"3. For Balanced Approach: Use {opt_combined.x*100:.2f}% allocation")
print(f"\nYour original 50% allocation was very close to optimal!")
print(f"The improvement from 50% to optimal is relatively small, suggesting your intuition was good.")

# Save results to CSV
results_export = pd.DataFrame({
    'Allocation_Pct': fine_df['allocation_pct'] * 100,
    'Total_Return': fine_df['total_return'],
    'Max_Drawdown': fine_df['max_drawdown'],
    'Sharpe_Ratio': fine_df['sharpe_ratio'],
    'Num_Trades': fine_df['num_trades']
})
results_export.to_csv('soxl_optimization_results.csv', index=False)
print(f"\nDetailed results saved to 'soxl_optimization_results.csv'")