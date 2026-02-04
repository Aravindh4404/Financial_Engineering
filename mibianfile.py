"""
Hedging Analysis - SIMPLE AND CORRECT VERSION
The key: Track CHANGES in value, not absolute values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

import mibian

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BlackScholesGreeks:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = max(T, 1e-10)
        self.r = r * 100
        self.sigma = sigma * 100
        self.option_type = option_type.lower()
        self.T_days = max(int(self.T * 365), 1)
        self.bs = mibian.BS([self.S, self.K, self.r, self.T_days], volatility=self.sigma)

    def price(self):
        if self.T <= 0:
            return max(0, self.S - self.K) if self.option_type == 'call' else max(0, self.K - self.S)
        return self.bs.callPrice if self.option_type == 'call' else self.bs.putPrice

    def delta(self):
        if self.T <= 0:
            return 1.0 if self.S > self.K else 0.0 if self.option_type == 'call' else (-1.0 if self.S < self.K else 0.0)
        return self.bs.callDelta if self.option_type == 'call' else self.bs.putDelta

    def gamma(self):
        return 0 if self.T <= 0 else self.bs.gamma

    def vega(self):
        return 0 if self.T <= 0 else self.bs.vega


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    if T <= 0:
        return 0.01
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if option_price <= intrinsic:
        return 0.01
    T_days = max(int(T * 365), 1)
    r_percent = r * 100
    try:
        if option_type == 'call':
            iv = mibian.BS([S, K, r_percent, T_days], callPrice=option_price).impliedVolatility
        else:
            iv = mibian.BS([S, K, r_percent, T_days], putPrice=option_price).impliedVolatibility
        return max(0.01, iv / 100)
    except:
        return 0.30


class HedgingSimulator:
    def __init__(self, data, strike, expiry_date, risk_free_rate=0.065):
        self.data = data.copy()
        self.strike = strike
        self.expiry_date = pd.to_datetime(expiry_date)
        self.r = risk_free_rate

        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%b-%Y')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        self.data['T'] = (self.expiry_date - self.data['Date']).dt.days / 365
        self.data['T'] = self.data['T'].clip(lower=0)

        self.data['IV'] = 0.0
        self.data['Delta'] = 0.0
        self.data['Gamma'] = 0.0
        self.data['Vega'] = 0.0

    def calculate_greeks(self):
        print("Calculating Greeks...")
        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = calculate_implied_volatility(row['Settle Price'], S, self.strike, T, self.r, 'call')
            self.data.at[idx, 'IV'] = iv

            bs = BlackScholesGreeks(S, self.strike, T, self.r, iv, 'call')
            self.data.at[idx, 'Delta'] = bs.delta()
            self.data.at[idx, 'Gamma'] = bs.gamma()
            self.data.at[idx, 'Vega'] = bs.vega()

        print("Done!")
        return self.data

    def simulate_unhedged(self, N=100):
        """
        Simple: We sold N calls
        P&L = Initial price - Current price (per contract) × N
        """
        results = []
        initial_price = self.data.iloc[0]['Settle Price']

        for idx, row in self.data.iterrows():
            current_price = row['Settle Price']
            # We're SHORT, so profit when price goes DOWN
            pnl_per_contract = initial_price - current_price
            total_pnl = pnl_per_contract * N

            results.append({
                'Date': row['Date'],
                'PnL': total_pnl
            })

        return pd.DataFrame(results)

    def simulate_delta_hedge(self, N=100, rebal_threshold=0.05):
        """
        We're short N calls, hedge with stock

        KEY INSIGHT: Track CUMULATIVE P&L from ALL sources
        """
        results = []

        # Initial values
        initial_option_price = self.data.iloc[0]['Settle Price']
        stock_shares = 0
        cumulative_pnl = 0
        last_rebal_delta = 0

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            delta = row['Delta']
            current_option_price = row['Settle Price']

            # STEP 1: P&L from option position (we're short)
            option_pnl_change = 0
            if idx > 0:
                prev_option_price = self.data.iloc[idx-1]['Settle Price']
                option_pnl_change = (prev_option_price - current_option_price) * N

            # STEP 2: P&L from stock position
            stock_pnl_change = 0
            if idx > 0:
                prev_S = self.data.iloc[idx-1]['Underlying Value']
                stock_pnl_change = (S - prev_S) * stock_shares

            # STEP 3: Rebalance if needed
            rebalanced = False
            rebal_cost = 0
            if idx == 0 or abs(delta - last_rebal_delta) > rebal_threshold:
                target_shares = delta * N
                shares_to_trade = abs(target_shares - stock_shares)
                rebal_cost = shares_to_trade * S * 0.0005  # 5 bps transaction cost
                stock_shares = target_shares
                last_rebal_delta = delta
                rebalanced = True

            # STEP 4: Update cumulative P&L
            cumulative_pnl += option_pnl_change + stock_pnl_change - rebal_cost

            results.append({
                'Date': row['Date'],
                'Stock_Shares': stock_shares,
                'PnL': cumulative_pnl,
                'Rebalanced': rebalanced
            })

        return pd.DataFrame(results)

    def simulate_delta_gamma_hedge(self, N=100):
        """
        Hedge with both stock AND another option
        """
        results = []
        hedge_strike = self.strike + 200

        initial_option_price = self.data.iloc[0]['Settle Price']
        stock_shares = 0
        hedge_contracts = 0
        cumulative_pnl = 0

        # Day 0: Set up hedge
        row0 = self.data.iloc[0]
        S0 = row0['Underlying Value']
        T0 = row0['T']
        iv0 = row0['IV']
        gamma0 = row0['Gamma']
        delta0 = row0['Delta']

        bs_hedge = BlackScholesGreeks(S0, hedge_strike, T0, self.r, iv0, 'call')
        hedge_price_0 = bs_hedge.price()
        hedge_gamma_0 = bs_hedge.gamma()
        hedge_delta_0 = bs_hedge.delta()

        # Buy hedge options (50% gamma neutralization)
        if hedge_gamma_0 > 0.00001:
            hedge_contracts = min((gamma0 * N * 0.5) / hedge_gamma_0, N * 1.5)
            # Cost of buying hedge options
            hedge_cost = hedge_contracts * hedge_price_0
            cumulative_pnl -= hedge_cost
            cumulative_pnl -= hedge_cost * 0.0005  # transaction cost

        # Stock position to delta hedge
        net_delta = delta0 * N - hedge_delta_0 * hedge_contracts
        stock_shares = net_delta
        stock_cost = abs(stock_shares * S0 * 0.0005)
        cumulative_pnl -= stock_cost

        # Track previous values
        prev_option_price = initial_option_price
        prev_hedge_price = hedge_price_0
        prev_S = S0

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            current_option_price = row['Settle Price']

            # Calculate current hedge option price
            bs_hedge = BlackScholesGreeks(S, hedge_strike, T, self.r, iv, 'call')
            current_hedge_price = bs_hedge.price()

            if idx > 0:
                # P&L from short call (we're short, profit when it goes down)
                option_pnl = (prev_option_price - current_option_price) * N

                # P&L from long hedge options (we're long, profit when it goes up)
                hedge_pnl = (current_hedge_price - prev_hedge_price) * hedge_contracts

                # P&L from stock
                stock_pnl = (S - prev_S) * stock_shares

                # Update cumulative
                cumulative_pnl += option_pnl + hedge_pnl + stock_pnl

                # Update prev values
                prev_option_price = current_option_price
                prev_hedge_price = current_hedge_price
                prev_S = S

            results.append({
                'Date': row['Date'],
                'PnL': cumulative_pnl
            })

        return pd.DataFrame(results)

    def simulate_vega_hedge(self, N=100):
        """Vega hedge with different strike"""
        results = []
        hedge_strike = self.strike + 500

        initial_option_price = self.data.iloc[0]['Settle Price']
        stock_shares = 0
        vega_hedge_contracts = 0
        cumulative_pnl = 0

        # Day 0 setup
        row0 = self.data.iloc[0]
        S0 = row0['Underlying Value']
        T0 = row0['T']
        iv0 = row0['IV']
        vega0 = row0['Vega']
        delta0 = row0['Delta']

        bs_hedge = BlackScholesGreeks(S0, hedge_strike, T0, self.r, iv0, 'call')
        hedge_price_0 = bs_hedge.price()
        hedge_vega_0 = bs_hedge.vega()
        hedge_delta_0 = bs_hedge.delta()

        # Buy vega hedge (50% neutralization)
        if hedge_vega_0 > 0.1:
            vega_hedge_contracts = min((vega0 * N * 0.5) / hedge_vega_0, N * 1.5)
            hedge_cost = vega_hedge_contracts * hedge_price_0
            cumulative_pnl -= hedge_cost
            cumulative_pnl -= hedge_cost * 0.0005

        # Delta hedge
        net_delta = delta0 * N - hedge_delta_0 * vega_hedge_contracts
        stock_shares = net_delta
        cumulative_pnl -= abs(stock_shares * S0 * 0.0005)

        prev_option_price = initial_option_price
        prev_hedge_price = hedge_price_0
        prev_S = S0

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            current_option_price = row['Settle Price']

            bs_hedge = BlackScholesGreeks(S, hedge_strike, T, self.r, iv, 'call')
            current_hedge_price = bs_hedge.price()

            if idx > 0:
                option_pnl = (prev_option_price - current_option_price) * N
                hedge_pnl = (current_hedge_price - prev_hedge_price) * vega_hedge_contracts
                stock_pnl = (S - prev_S) * stock_shares

                cumulative_pnl += option_pnl + hedge_pnl + stock_pnl

                prev_option_price = current_option_price
                prev_hedge_price = current_hedge_price
                prev_S = S

            results.append({
                'Date': row['Date'],
                'PnL': cumulative_pnl
            })

        return pd.DataFrame(results)


def create_visualizations(data, unhedged, delta_hedged, delta_gamma_hedged, vega_hedged):
    fig = plt.figure(figsize=(18, 12))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # 1. P&L Comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(unhedged['Date'], unhedged['PnL'], label='Unhedged', linewidth=2, color=colors[0])
    ax1.plot(delta_hedged['Date'], delta_hedged['PnL'], label='Delta Hedged', linewidth=2, color=colors[1])
    ax1.plot(delta_gamma_hedged['Date'], delta_gamma_hedged['PnL'], label='Delta-Gamma', linewidth=2, color=colors[2])
    ax1.plot(vega_hedged['Date'], vega_hedged['PnL'], label='Vega Hedged', linewidth=2, color=colors[3])
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('P&L (₹)')
    ax1.set_title('P&L Comparison (Correct Accounting)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Delta
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(data['Date'], data['Delta'], linewidth=2)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Delta')
    ax2.set_title('Delta', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Gamma
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(data['Date'], data['Gamma'], linewidth=2, color='orange')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Gamma')
    ax3.set_title('Gamma', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # 4. IV
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(data['Date'], data['IV'] * 100, linewidth=2, color='purple')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('IV (%)')
    ax4.set_title('Implied Volatility', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # 5. Vega
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(data['Date'], data['Vega'], linewidth=2, color='green')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Vega')
    ax5.set_title('Vega', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

    # 6. Underlying vs Option
    ax6 = plt.subplot(3, 3, 6)
    ax6_twin = ax6.twinx()
    ax6.plot(data['Date'], data['Underlying Value'], linewidth=2, color='blue')
    ax6_twin.plot(data['Date'], data['Settle Price'], linewidth=2, color='red')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('NIFTY', color='blue')
    ax6_twin.set_ylabel('Option', color='red')
    ax6.set_title('Prices', fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

    # 7. Delta Hedge Position
    ax7 = plt.subplot(3, 3, 7)
    rebalances = delta_hedged[delta_hedged['Rebalanced']]['Date']
    ax7.plot(delta_hedged['Date'], delta_hedged['Stock_Shares'], linewidth=2)
    ax7.scatter(rebalances, delta_hedged[delta_hedged['Rebalanced']]['Stock_Shares'],
                color='red', s=50, zorder=5)
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Shares')
    ax7.set_title('Delta Hedge: Stock Position', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45)

    # 8. Risk
    ax8 = plt.subplot(3, 3, 8)
    strategies = ['Unhedged', 'Delta\nHedged', 'Delta-Gamma\nHedged', 'Vega\nHedged']
    risk = [
        unhedged['PnL'].std(),
        delta_hedged['PnL'].std(),
        delta_gamma_hedged['PnL'].std(),
        vega_hedged['PnL'].std()
    ]
    ax8.bar(strategies, risk, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Std Dev (₹)')
    ax8.set_title('Risk (should decrease with hedging)', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Final P&L
    ax9 = plt.subplot(3, 3, 9)
    final_pnl = [
        unhedged['PnL'].iloc[-1],
        delta_hedged['PnL'].iloc[-1],
        delta_gamma_hedged['PnL'].iloc[-1],
        vega_hedged['PnL'].iloc[-1]
    ]
    colors_final = ['red' if x < 0 else 'green' for x in final_pnl]
    ax9.bar(strategies, final_pnl, color=colors_final, alpha=0.7, edgecolor='black')
    ax9.set_ylabel('Final P&L (₹)')
    ax9.set_title('Final P&L (hedging may reduce profit)', fontweight='bold')
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs('ALL Output', exist_ok=True)
    plt.savefig('ALL Output/hedging_final_correct.png', dpi=300, bbox_inches='tight')
    print("Saved visualization!")


def main():
    print("="*80)
    print("HEDGING ANALYSIS - FINAL CORRECT VERSION")
    print("="*80)
    print()

    df = pd.read_csv('Data/OPTIDX_NIFTY_CE_04-Nov-2025_TO_04-Feb-2026.csv')
    df.columns = df.columns.str.strip()
    df = df[['Date', 'Settle Price', 'Underlying Value', 'Strike Price', 'Expiry']].copy()

    print(f"Loaded {len(df)} records")

    simulator = HedgingSimulator(df, df['Strike Price'].iloc[0], df['Expiry'].iloc[0])
    data = simulator.calculate_greeks()

    print("\nRunning simulations...")
    unhedged = simulator.simulate_unhedged(100)
    delta_hedged = simulator.simulate_delta_hedge(100)
    delta_gamma = simulator.simulate_delta_gamma_hedge(100)
    vega_hedge = simulator.simulate_vega_hedge(100)

    summary = pd.DataFrame({
        'Strategy': ['Unhedged', 'Delta Hedged', 'Delta-Gamma', 'Vega Hedged'],
        'Final P&L': [
            unhedged['PnL'].iloc[-1],
            delta_hedged['PnL'].iloc[-1],
            delta_gamma['PnL'].iloc[-1],
            vega_hedge['PnL'].iloc[-1]
        ],
        'Risk (Std Dev)': [
            unhedged['PnL'].std(),
            delta_hedged['PnL'].std(),
            delta_gamma['PnL'].std(),
            vega_hedge['PnL'].std()
        ]
    })

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(summary.round(0).to_string(index=False))
    print("\n✓ Risk should DECREASE with hedging")
    print("✓ Profit may also DECREASE (that's the cost of insurance)")

    create_visualizations(data, unhedged, delta_hedged, delta_gamma, vega_hedge)

    os.makedirs('ALL Output', exist_ok=True)
    summary.to_csv('ALL Output/summary_final.csv', index=False)


if __name__ == "__main__":
    main()