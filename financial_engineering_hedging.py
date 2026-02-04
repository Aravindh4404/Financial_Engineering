"""
Financial Engineering: Delta, Delta-Gamma, and Vega Hedging Analysis
DATA 609 - Group Project

This script demonstrates hedging strategies using real NIFTY options data:
1. Delta Hedging
2. Delta-Gamma Hedging
3. Vega Hedging
4. Comparison of P&L across strategies
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BlackScholesGreeks:
    """
    Black-Scholes model for European options pricing and Greeks calculation
    """
    
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        S: Current underlying price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility (implied volatility)
        option_type: 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
    def d1(self):
        """Calculate d1 parameter"""
        if self.T <= 0:
            return 0
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        """Calculate d2 parameter"""
        if self.T <= 0:
            return 0
        return self.d1() - self.sigma * np.sqrt(self.T)
    
    def price(self):
        """Calculate option price"""
        if self.T <= 0:
            return max(0, self.S - self.K) if self.option_type == 'call' else max(0, self.K - self.S)
        
        d1 = self.d1()
        d2 = self.d2()
        
        if self.option_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def delta(self):
        """Calculate delta"""
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        
        d1 = self.d1()
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return -norm.cdf(-d1)
    
    def gamma(self):
        """Calculate gamma"""
        if self.T <= 0:
            return 0
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Calculate vega (sensitivity to 1% change in volatility)"""
        if self.T <= 0:
            return 0
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100  # Divided by 100 for 1% change
    
    def theta(self):
        """Calculate theta (time decay per day)"""
        if self.T <= 0:
            return 0
        d1 = self.d1()
        d2 = self.d2()
        
        if self.option_type == 'call':
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        return theta / 365  # Per day


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call', 
                                 initial_guess=0.3, max_iterations=100):
    """
    Calculate implied volatility using Newton-Raphson method
    """
    if T <= 0:
        return 0
    
    # Intrinsic value check
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if option_price <= intrinsic:
        return 0.01  # Minimum volatility
    
    def objective(sigma):
        """Objective function to minimize"""
        try:
            bs = BlackScholesGreeks(S, K, T, r, sigma, option_type)
            return abs(bs.price() - option_price)
        except:
            return 1e10
    
    # Use scipy's minimize_scalar for robust optimization
    result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
    
    if result.success:
        return result.x
    else:
        return initial_guess


class HedgingSimulator:
    """
    Simulate various hedging strategies for options portfolios
    """
    
    def __init__(self, data, strike, expiry_date, risk_free_rate=0.065):
        """
        data: DataFrame with columns [Date, Settle Price, Underlying Value]
        strike: Strike price of the option
        expiry_date: Expiry date of the option
        risk_free_rate: Annual risk-free rate (default 6.5% for India)
        """
        self.data = data.copy()
        self.strike = strike
        self.expiry_date = pd.to_datetime(expiry_date)
        self.r = risk_free_rate
        
        # Sort by date
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%b-%Y')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Calculate time to expiry for each date
        self.data['T'] = (self.expiry_date - self.data['Date']).dt.days / 365
        self.data['T'] = self.data['T'].clip(lower=0)
        
        # Initialize Greeks columns
        self.data['IV'] = 0.0
        self.data['Delta'] = 0.0
        self.data['Gamma'] = 0.0
        self.data['Vega'] = 0.0
        self.data['Theta'] = 0.0
        
    def calculate_greeks(self):
        """Calculate implied volatility and Greeks for all dates"""
        print("Calculating Implied Volatility and Greeks...")
        
        for idx, row in self.data.iterrows():
            option_price = row['Settle Price']
            S = row['Underlying Value']
            T = row['T']
            
            # Calculate IV
            iv = calculate_implied_volatility(option_price, S, self.strike, T, self.r, 'call')
            self.data.at[idx, 'IV'] = iv
            
            # Calculate Greeks
            bs = BlackScholesGreeks(S, self.strike, T, self.r, iv, 'call')
            self.data.at[idx, 'Delta'] = bs.delta()
            self.data.at[idx, 'Gamma'] = bs.gamma()
            self.data.at[idx, 'Vega'] = bs.vega()
            self.data.at[idx, 'Theta'] = bs.theta()
        
        print("Greeks calculation complete!")
        return self.data
    
    def simulate_unhedged(self, num_options_sold=100):
        """
        Simulate an unhedged short call position
        Returns: DataFrame with daily P&L
        """
        results = []
        
        # Initial position: Sell call options
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        
        for idx, row in self.data.iterrows():
            option_value = row['Settle Price'] * num_options_sold
            pnl = initial_premium - option_value  # Profit if option value decreases
            
            results.append({
                'Date': row['Date'],
                'Option_Value': option_value,
                'PnL': pnl,
                'Cumulative_PnL': pnl
            })
        
        return pd.DataFrame(results)
    
    def simulate_delta_hedge(self, num_options_sold=100, rebalance_threshold=0.05):
        """
        Simulate delta hedging by buying/selling underlying
        
        Strategy: 
        - Sell call options (short position)
        - Buy underlying stock to neutralize delta
        - Rebalance when delta changes significantly
        
        rebalance_threshold: Rebalance when delta change exceeds this threshold
        """
        results = []
        
        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        cash = initial_premium  # Cash from selling options
        stock_position = 0  # Number of underlying shares
        last_rebalance_delta = 0
        
        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            delta = row['Delta']
            option_value = row['Settle Price'] * num_options_sold
            
            # Target stock position to neutralize delta (negative because we're short calls)
            target_stock = -delta * num_options_sold
            
            # Rebalance if threshold exceeded or first day
            if idx == 0 or abs(delta - last_rebalance_delta) > rebalance_threshold:
                stock_to_trade = target_stock - stock_position
                cash -= stock_to_trade * S  # Buy/sell stock
                stock_position = target_stock
                last_rebalance_delta = delta
                rebalanced = True
            else:
                rebalanced = False
            
            # Calculate portfolio value
            portfolio_value = cash + stock_position * S - option_value
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Cash': cash,
                'Option_Value': option_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl,
                'Rebalanced': rebalanced,
                'Delta': delta
            })
        
        return pd.DataFrame(results)
    
    def simulate_delta_gamma_hedge(self, num_options_sold=100, hedge_strike_offset=200):
        """
        Simulate delta-gamma hedging using another option
        
        Strategy:
        - Sell call options at strike K (short position)
        - Buy another option at strike K+offset to hedge gamma
        - Buy/sell underlying to neutralize remaining delta
        
        hedge_strike_offset: Strike difference for hedging option
        """
        results = []
        hedge_strike = self.strike + hedge_strike_offset
        
        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        cash = initial_premium
        stock_position = 0
        hedge_option_position = 0  # Number of hedge options bought
        
        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            
            # Calculate Greeks for main option (already have)
            delta_main = row['Delta']
            gamma_main = row['Gamma']
            option_value = row['Settle Price'] * num_options_sold
            
            # Calculate Greeks for hedge option (ATM or slightly OTM)
            bs_hedge = BlackScholesGreeks(S, hedge_strike, T, self.r, iv, 'call')
            delta_hedge = bs_hedge.delta()
            gamma_hedge = bs_hedge.gamma()
            hedge_option_price = bs_hedge.price()
            
            # First day: establish positions
            if idx == 0:
                # Neutralize gamma: gamma_main * N + gamma_hedge * N_hedge = 0
                # Since we're short N options: -gamma_main * N + gamma_hedge * N_hedge = 0
                if gamma_hedge > 0:
                    hedge_option_position = (gamma_main * num_options_sold) / gamma_hedge
                    cash -= hedge_option_position * hedge_option_price  # Buy hedge options
                
                # Neutralize remaining delta
                portfolio_delta = -delta_main * num_options_sold + delta_hedge * hedge_option_position
                stock_position = -portfolio_delta
                cash -= stock_position * S
            
            # Calculate current portfolio value
            hedge_option_value = hedge_option_position * hedge_option_price
            portfolio_value = (cash + stock_position * S + 
                              hedge_option_value - option_value)
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Hedge_Option_Position': hedge_option_position,
                'Cash': cash,
                'Option_Value': option_value,
                'Hedge_Option_Value': hedge_option_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl,
                'Gamma_Main': gamma_main,
                'Gamma_Hedge': gamma_hedge,
                'Delta_Main': delta_main,
                'Delta_Hedge': delta_hedge
            })
        
        return pd.DataFrame(results)
    
    def simulate_vega_hedge(self, num_options_sold=100, hedge_strike_offset=500):
        """
        Simulate vega hedging to protect against volatility changes
        
        Strategy:
        - Sell call options (short vega position)
        - Buy options with different strikes to hedge vega
        - Manage delta with underlying
        """
        results = []
        hedge_strike = self.strike + hedge_strike_offset
        
        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        cash = initial_premium
        stock_position = 0
        vega_hedge_position = 0
        
        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            
            # Main option Greeks
            vega_main = row['Vega']
            delta_main = row['Delta']
            option_value = row['Settle Price'] * num_options_sold
            
            # Vega hedge option Greeks
            bs_vega_hedge = BlackScholesGreeks(S, hedge_strike, T, self.r, iv, 'call')
            vega_hedge = bs_vega_hedge.vega()
            delta_vega_hedge = bs_vega_hedge.delta()
            vega_hedge_price = bs_vega_hedge.price()
            
            # First day: establish positions
            if idx == 0:
                # Neutralize vega: -vega_main * N + vega_hedge * N_vega = 0
                if vega_hedge > 0:
                    vega_hedge_position = (vega_main * num_options_sold) / vega_hedge
                    cash -= vega_hedge_position * vega_hedge_price
                
                # Neutralize delta
                portfolio_delta = -delta_main * num_options_sold + delta_vega_hedge * vega_hedge_position
                stock_position = -portfolio_delta
                cash -= stock_position * S
            
            # Portfolio value
            vega_hedge_value = vega_hedge_position * vega_hedge_price
            portfolio_value = (cash + stock_position * S + 
                              vega_hedge_value - option_value)
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Vega_Hedge_Position': vega_hedge_position,
                'Cash': cash,
                'Option_Value': option_value,
                'Vega_Hedge_Value': vega_hedge_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl,
                'Vega_Main': vega_main,
                'Vega_Hedge': vega_hedge,
                'IV': iv
            })
        
        return pd.DataFrame(results)


def create_visualizations(data, unhedged, delta_hedged, delta_gamma_hedged, vega_hedged):
    """
    Create comprehensive visualizations for the report
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Greeks Evolution
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(data['Date'], data['Delta'], label='Delta', linewidth=2)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Delta', fontsize=10)
    ax1.set_title('Delta Evolution Over Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(data['Date'], data['Gamma'], label='Gamma', color='orange', linewidth=2)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Gamma', fontsize=10)
    ax2.set_title('Gamma Evolution Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(data['Date'], data['Vega'], label='Vega', color='green', linewidth=2)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Vega', fontsize=10)
    ax3.set_title('Vega Evolution Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 2. Implied Volatility
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(data['Date'], data['IV'] * 100, label='Implied Volatility', color='purple', linewidth=2)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('IV (%)', fontsize=10)
    ax4.set_title('Implied Volatility Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 3. P&L Comparison - Main Chart
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(unhedged['Date'], unhedged['PnL'], label='Unhedged', linewidth=2, alpha=0.8)
    ax5.plot(delta_hedged['Date'], delta_hedged['PnL'], label='Delta Hedged', linewidth=2, alpha=0.8)
    ax5.plot(delta_gamma_hedged['Date'], delta_gamma_hedged['PnL'], label='Delta-Gamma Hedged', linewidth=2, alpha=0.8)
    ax5.plot(vega_hedged['Date'], vega_hedged['PnL'], label='Vega Hedged', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('P&L (₹)', fontsize=10)
    ax5.set_title('P&L Comparison: All Strategies', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Underlying Price and Option Price
    ax6 = plt.subplot(3, 3, 6)
    ax6_twin = ax6.twinx()
    ax6.plot(data['Date'], data['Underlying Value'], label='NIFTY Price', color='blue', linewidth=2)
    ax6_twin.plot(data['Date'], data['Settle Price'], label='Option Price', color='red', linewidth=2)
    ax6.set_xlabel('Date', fontsize=10)
    ax6.set_ylabel('NIFTY Price (₹)', fontsize=10, color='blue')
    ax6_twin.set_ylabel('Option Price (₹)', fontsize=10, color='red')
    ax6.set_title('Underlying vs Option Price', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # 5. Delta Hedge Rebalancing
    ax7 = plt.subplot(3, 3, 7)
    rebalance_dates = delta_hedged[delta_hedged['Rebalanced']]['Date']
    ax7.plot(delta_hedged['Date'], delta_hedged['Stock_Position'], linewidth=2)
    ax7.scatter(rebalance_dates, delta_hedged[delta_hedged['Rebalanced']]['Stock_Position'], 
                color='red', s=50, zorder=5, label='Rebalance Points')
    ax7.set_xlabel('Date', fontsize=10)
    ax7.set_ylabel('Stock Position', fontsize=10)
    ax7.set_title('Delta Hedge: Stock Position & Rebalancing', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45)
    
    # 6. P&L Volatility (Risk)
    ax8 = plt.subplot(3, 3, 8)
    strategies = ['Unhedged', 'Delta\nHedged', 'Delta-Gamma\nHedged', 'Vega\nHedged']
    pnl_std = [
        unhedged['PnL'].std(),
        delta_hedged['PnL'].std(),
        delta_gamma_hedged['PnL'].std(),
        vega_hedged['PnL'].std()
    ]
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax8.bar(strategies, pnl_std, color=colors_bar, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('P&L Standard Deviation (₹)', fontsize=10)
    ax8.set_title('Risk Comparison: P&L Volatility', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 7. Final P&L Comparison
    ax9 = plt.subplot(3, 3, 9)
    final_pnl = [
        unhedged['PnL'].iloc[-1],
        delta_hedged['PnL'].iloc[-1],
        delta_gamma_hedged['PnL'].iloc[-1],
        vega_hedged['PnL'].iloc[-1]
    ]
    colors_final = ['red' if x < 0 else 'green' for x in final_pnl]
    ax9.bar(strategies, final_pnl, color=colors_final, alpha=0.7, edgecolor='black')
    ax9.set_ylabel('Final P&L (₹)', fontsize=10)
    ax9.set_title('Final P&L Comparison', fontsize=12, fontweight='bold')
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join('ALL Output', 'hedging_analysis_complete.png')
    os.makedirs('ALL Output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Comprehensive visualization saved!")
    
    return fig


def generate_summary_statistics(data, unhedged, delta_hedged, delta_gamma_hedged, vega_hedged):
    """
    Generate summary statistics for report
    """
    summary = {
        'Strategy': ['Unhedged', 'Delta Hedged', 'Delta-Gamma Hedged', 'Vega Hedged'],
        'Final P&L (₹)': [
            unhedged['PnL'].iloc[-1],
            delta_hedged['PnL'].iloc[-1],
            delta_gamma_hedged['PnL'].iloc[-1],
            vega_hedged['PnL'].iloc[-1]
        ],
        'Max P&L (₹)': [
            unhedged['PnL'].max(),
            delta_hedged['PnL'].max(),
            delta_gamma_hedged['PnL'].max(),
            vega_hedged['PnL'].max()
        ],
        'Min P&L (₹)': [
            unhedged['PnL'].min(),
            delta_hedged['PnL'].min(),
            delta_gamma_hedged['PnL'].min(),
            vega_hedged['PnL'].min()
        ],
        'P&L Std Dev (₹)': [
            unhedged['PnL'].std(),
            delta_hedged['PnL'].std(),
            delta_gamma_hedged['PnL'].std(),
            vega_hedged['PnL'].std()
        ],
        'Sharpe Ratio': [
            unhedged['PnL'].mean() / unhedged['PnL'].std() if unhedged['PnL'].std() > 0 else 0,
            delta_hedged['PnL'].mean() / delta_hedged['PnL'].std() if delta_hedged['PnL'].std() > 0 else 0,
            delta_gamma_hedged['PnL'].mean() / delta_gamma_hedged['PnL'].std() if delta_gamma_hedged['PnL'].std() > 0 else 0,
            vega_hedged['PnL'].mean() / vega_hedged['PnL'].std() if vega_hedged['PnL'].std() > 0 else 0
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.round(2)
    
    return summary_df


def main():
    """
    Main execution function
    """
    print("="*80)
    print("FINANCIAL ENGINEERING: HEDGING STRATEGIES ANALYSIS")
    print("DATA 609 - Group Project")
    print("="*80)
    print()
    
    # Load data
    print("Loading NIFTY options data...")
    data_file = os.path.join('Data', 'OPTIDX_NIFTY_CE_04-Nov-2025_TO_04-Feb-2026.csv')
    df = pd.read_csv(data_file)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # Select relevant columns
    df = df[['Date', 'Settle Price', 'Underlying Value', 'Strike Price', 'Expiry']].copy()
    
    # Basic data info
    print(f"Data loaded: {len(df)} records")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Strike Price: {df['Strike Price'].iloc[0]}")
    print(f"Expiry: {df['Expiry'].iloc[0]}")
    print()
    
    # Initialize simulator
    strike = df['Strike Price'].iloc[0]
    expiry = df['Expiry'].iloc[0]
    
    simulator = HedgingSimulator(df, strike, expiry)
    
    # Calculate Greeks
    data_with_greeks = simulator.calculate_greeks()
    
    print("\nGreeks Summary (Latest Date):")
    print(f"  Delta: {data_with_greeks['Delta'].iloc[-1]:.4f}")
    print(f"  Gamma: {data_with_greeks['Gamma'].iloc[-1]:.6f}")
    print(f"  Vega: {data_with_greeks['Vega'].iloc[-1]:.4f}")
    print(f"  Implied Vol: {data_with_greeks['IV'].iloc[-1]*100:.2f}%")
    print()
    
    # Run simulations
    print("Running hedging simulations...")
    print("  1. Unhedged position...")
    unhedged = simulator.simulate_unhedged(num_options_sold=100)
    
    print("  2. Delta hedging...")
    delta_hedged = simulator.simulate_delta_hedge(num_options_sold=100)
    
    print("  3. Delta-gamma hedging...")
    delta_gamma_hedged = simulator.simulate_delta_gamma_hedge(num_options_sold=100)
    
    print("  4. Vega hedging...")
    vega_hedged = simulator.simulate_vega_hedge(num_options_sold=100)
    
    print("\nSimulations complete!")
    print()
    
    # Generate summary statistics
    print("Generating summary statistics...")
    summary = generate_summary_statistics(data_with_greeks, unhedged, delta_hedged, 
                                          delta_gamma_hedged, vega_hedged)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary.to_string(index=False))
    print()
    
    # Create output directory if it doesn't exist
    output_dir = 'ALL Output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'hedging_summary_statistics.csv')
    summary.to_csv(summary_path, index=False)
    print("Summary statistics saved to: hedging_summary_statistics.csv")
    
    # Save detailed results
    data_with_greeks.to_csv(os.path.join(output_dir, 'greeks_data.csv'), index=False)
    unhedged.to_csv(os.path.join(output_dir, 'unhedged_pnl.csv'), index=False)
    delta_hedged.to_csv(os.path.join(output_dir, 'delta_hedged_pnl.csv'), index=False)
    delta_gamma_hedged.to_csv(os.path.join(output_dir, 'delta_gamma_hedged_pnl.csv'), index=False)
    vega_hedged.to_csv(os.path.join(output_dir, 'vega_hedged_pnl.csv'), index=False)
    
    print("\nDetailed results saved:")
    print("  - greeks_data.csv")
    print("  - unhedged_pnl.csv")
    print("  - delta_hedged_pnl.csv")
    print("  - delta_gamma_hedged_pnl.csv")
    print("  - vega_hedged_pnl.csv")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(data_with_greeks, unhedged, delta_hedged, 
                         delta_gamma_hedged, vega_hedged)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"  • Unhedged P&L: ₹{unhedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Delta Hedged P&L: ₹{delta_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Delta-Gamma Hedged P&L: ₹{delta_gamma_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Vega Hedged P&L: ₹{vega_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"\n  • Risk Reduction (Delta Hedge): {(1 - delta_hedged['PnL'].std()/unhedged['PnL'].std())*100:.1f}%")
    print(f"  • Risk Reduction (Delta-Gamma): {(1 - delta_gamma_hedged['PnL'].std()/unhedged['PnL'].std())*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
