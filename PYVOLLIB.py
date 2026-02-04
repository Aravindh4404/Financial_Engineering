"""
Financial Engineering: Delta, Delta-Gamma, and Vega Hedging Analysis
DATA 609 - Group Project

This script demonstrates hedging strategies using real NIFTY options data:
1. Delta Hedging
2. Delta-Gamma Hedging
3. Vega Hedging
4. Comparison of P&L across strategies

Simplified version using py_vollib for Black-Scholes calculations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings('ignore')

# Import py_vollib for option pricing and Greeks
from py_vollib.black_scholes import black_scholes as bs_price
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta
from py_vollib.black_scholes.implied_volatility import implied_volatility

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BlackScholesGreeks:
    """
    Simplified Black-Scholes wrapper using py_vollib
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        S: Current underlying price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility (implied volatility)
        option_type: 'call' or 'put' (must be lowercase for py_vollib)
        """
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)  # Avoid zero time
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

    def price(self):
        """Calculate option price using py_vollib"""
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)

        flag = 'c' if self.option_type == 'call' else 'p'
        return bs_price(flag, self.S, self.K, self.T, self.r, self.sigma)

    def delta(self):
        """Calculate delta using py_vollib"""
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0

        flag = 'c' if self.option_type == 'call' else 'p'
        return delta(flag, self.S, self.K, self.T, self.r, self.sigma)

    def gamma(self):
        """Calculate gamma using py_vollib"""
        if self.T <= 0:
            return 0
        return gamma(flag='c', S=self.S, K=self.K, t=self.T, r=self.r, sigma=self.sigma)

    def vega(self):
        """Calculate vega (sensitivity to 1% change in volatility) using py_vollib"""
        if self.T <= 0:
            return 0
        # py_vollib returns vega for 100% change, so divide by 100 for 1% change
        return vega(flag='c', S=self.S, K=self.K, t=self.T, r=self.r, sigma=self.sigma) / 100

    def theta(self):
        """Calculate theta (time decay per day) using py_vollib"""
        if self.T <= 0:
            return 0
        flag = 'c' if self.option_type == 'call' else 'p'
        # py_vollib returns theta per year, divide by 365 for per day
        return theta(flag, self.S, self.K, self.T, self.r, self.sigma) / 365


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    """
    Calculate implied volatility using py_vollib
    """
    if T <= 0:
        return 0.01

    # Intrinsic value check
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if option_price <= intrinsic:
        return 0.01  # Minimum volatility

    flag = 'c' if option_type == 'call' else 'p'

    try:
        # Use py_vollib's implied_volatility function
        iv = implied_volatility(option_price, S, K, T, r, flag)
        return max(0.01, iv)  # Ensure positive volatility
    except Exception as e:
        print(f"Warning: IV calculation failed for price={option_price}, S={S}, K={K}, T={T}")
        return 0.30  # Return default volatility


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
        print("Calculating Implied Volatility and Greeks using py_vollib...")

        for idx, row in self.data.iterrows():
            option_price = row['Settle Price']
            S = row['Underlying Value']
            T = row['T']

            # Calculate IV
            iv = calculate_implied_volatility(option_price, S, self.strike, T, self.r, 'call')
            self.data.at[idx, 'IV'] = iv

            # Calculate Greeks using py_vollib
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

    def simulate_delta_hedge(self, num_options_sold=100, rebalance_threshold=0.1):
        """
        Simulate delta hedging strategy
        Rebalance when delta changes by more than threshold
        """
        results = []

        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        stock_position = 0  # Start with no stock position
        cash_balance = initial_premium
        previous_delta = 0

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            option_value = row['Settle Price'] * num_options_sold
            delta_position = row['Delta'] * num_options_sold

            # Check if rebalancing is needed
            delta_change = abs(delta_position - previous_delta)
            rebalanced = (delta_change > rebalance_threshold) or (idx == 0)

            if rebalanced:
                # Adjust stock position to match delta
                stock_to_trade = delta_position - stock_position
                cash_balance -= stock_to_trade * S  # Buy/sell stock
                stock_position = delta_position
                previous_delta = delta_position

            # Calculate portfolio value
            portfolio_value = cash_balance + stock_position * S - option_value
            pnl = portfolio_value - initial_premium

            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Option_Value': option_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl,
                'Rebalanced': rebalanced
            })

        return pd.DataFrame(results)

    def simulate_delta_gamma_hedge(self, num_options_sold=100):
        """
        Simulate delta-gamma hedging using additional options
        """
        results = []

        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        stock_position = 0
        hedge_option_position = 0  # Additional options for gamma hedging
        cash_balance = initial_premium

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            option_value = row['Settle Price'] * num_options_sold

            # Calculate required positions
            target_delta = row['Delta'] * num_options_sold
            target_gamma = row['Gamma'] * num_options_sold

            # Use additional options to hedge gamma (simplified)
            # In practice, you'd use options at different strikes
            hedge_option_gamma = row['Gamma'] * 0.5  # Assume hedge option has half the gamma
            hedge_options_needed = -target_gamma / hedge_option_gamma if hedge_option_gamma != 0 else 0

            # Adjust positions
            if idx == 0 or abs(hedge_options_needed - hedge_option_position) > 10:
                hedge_option_position = hedge_options_needed
                hedge_option_cost = hedge_option_position * row['Settle Price']
                cash_balance -= hedge_option_cost

            # Delta hedge the combined position
            combined_delta = target_delta + (hedge_option_position * row['Delta'])
            stock_position = combined_delta

            # Calculate portfolio value
            hedge_option_value = hedge_option_position * row['Settle Price']
            portfolio_value = cash_balance + stock_position * S - option_value + hedge_option_value
            pnl = portfolio_value - initial_premium

            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Hedge_Options': hedge_option_position,
                'Option_Value': option_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl
            })

        return pd.DataFrame(results)

    def simulate_vega_hedge(self, num_options_sold=100):
        """
        Simulate vega hedging to manage volatility risk
        """
        results = []

        # Initial setup
        initial_premium = self.data.iloc[0]['Settle Price'] * num_options_sold
        stock_position = 0
        vega_hedge_position = 0  # Options for vega hedging
        cash_balance = initial_premium

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            option_value = row['Settle Price'] * num_options_sold

            # Calculate vega exposure
            target_vega = row['Vega'] * num_options_sold

            # Use additional options to hedge vega
            hedge_option_vega = row['Vega'] * 0.6  # Assume hedge option vega
            vega_hedge_needed = -target_vega / hedge_option_vega if hedge_option_vega != 0 else 0

            # Adjust vega hedge position
            if idx == 0 or abs(vega_hedge_needed - vega_hedge_position) > 10:
                vega_hedge_position = vega_hedge_needed
                vega_hedge_cost = vega_hedge_position * row['Settle Price']
                cash_balance -= vega_hedge_cost

            # Delta hedge the combined position
            combined_delta = row['Delta'] * num_options_sold + vega_hedge_position * row['Delta']
            stock_position = combined_delta

            # Calculate portfolio value
            vega_hedge_value = vega_hedge_position * row['Settle Price']
            portfolio_value = cash_balance + stock_position * S - option_value + vega_hedge_value
            pnl = portfolio_value - initial_premium

            results.append({
                'Date': row['Date'],
                'Stock_Position': stock_position,
                'Vega_Hedge_Options': vega_hedge_position,
                'Option_Value': option_value,
                'Portfolio_Value': portfolio_value,
                'PnL': pnl
            })

        return pd.DataFrame(results)


def create_visualizations(data, unhedged, delta_hedged, delta_gamma_hedged, vega_hedged):
    """
    Create comprehensive visualization dashboard
    """
    fig = plt.figure(figsize=(18, 12))

    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # 1. Cumulative P&L Comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(unhedged['Date'], unhedged['PnL'], label='Unhedged',
             linewidth=2, color=colors[0])
    ax1.plot(delta_hedged['Date'], delta_hedged['PnL'], label='Delta Hedged',
             linewidth=2, color=colors[1])
    ax1.plot(delta_gamma_hedged['Date'], delta_gamma_hedged['PnL'], label='Delta-Gamma Hedged',
             linewidth=2, color=colors[2])
    ax1.plot(vega_hedged['Date'], vega_hedged['PnL'], label='Vega Hedged',
             linewidth=2, color=colors[3])
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('P&L (₹)', fontsize=10)
    ax1.set_title('Cumulative P&L Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Greeks Over Time
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(data['Date'], data['Delta'], label='Delta', linewidth=2)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Delta', fontsize=10)
    ax2.set_title('Delta Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(data['Date'], data['Gamma'], label='Gamma', linewidth=2, color='orange')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Gamma', fontsize=10)
    ax3.set_title('Gamma Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 3. Implied Volatility
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(data['Date'], data['IV'] * 100, linewidth=2, color='purple')
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax4.set_title('Implied Volatility Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # 4. Vega Over Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(data['Date'], data['Vega'], linewidth=2, color='green')
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Vega', fontsize=10)
    ax5.set_title('Vega Over Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

    # 5. Underlying Price vs Option Price
    ax6 = plt.subplot(3, 3, 6)
    ax6_twin = ax6.twinx()
    ax6.plot(data['Date'], data['Underlying Value'], linewidth=2, color='blue', label='NIFTY')
    ax6_twin.plot(data['Date'], data['Settle Price'], linewidth=2, color='red', label='Option')
    ax6.set_xlabel('Date', fontsize=10)
    ax6.set_ylabel('NIFTY Price (₹)', fontsize=10, color='blue')
    ax6_twin.set_ylabel('Option Price (₹)', fontsize=10, color='red')
    ax6.set_title('Underlying vs Option Price', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

    # 6. Delta Hedge Rebalancing
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

    # 7. P&L Volatility (Risk)
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

    # 8. Final P&L Comparison
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
            delta_gamma_hedged['PnL'].mean() / delta_gamma_hedged['PnL'].std() if delta_gamma_hedged[
                                                                                      'PnL'].std() > 0 else 0,
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
    print("=" * 80)
    print("FINANCIAL ENGINEERING: HEDGING STRATEGIES ANALYSIS")
    print("DATA 609 - Group Project")
    print("Using py_vollib for Black-Scholes calculations")
    print("=" * 80)
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
    print(f"  Implied Vol: {data_with_greeks['IV'].iloc[-1] * 100:.2f}%")
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

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
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

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  • Unhedged P&L: ₹{unhedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Delta Hedged P&L: ₹{delta_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Delta-Gamma Hedged P&L: ₹{delta_gamma_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"  • Vega Hedged P&L: ₹{vega_hedged['PnL'].iloc[-1]:,.2f}")
    print(f"\n  • Risk Reduction (Delta Hedge): {(1 - delta_hedged['PnL'].std() / unhedged['PnL'].std()) * 100:.1f}%")
    print(
        f"  • Risk Reduction (Delta-Gamma): {(1 - delta_gamma_hedged['PnL'].std() / unhedged['PnL'].std()) * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()