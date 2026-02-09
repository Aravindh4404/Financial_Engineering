"""
Financial Engineering Hedging Analysis - IMPROVED VERSION WITH COST BREAKDOWN
For DATA 609 Project Report

NEW FEATURES:
- Separate tracking of hedging costs
- Breakdown of P&L components (option P&L, hedge P&L, transaction costs)
- Clear visualization showing why hedging appears to "lose money"
- Comparison of protected vs unprotected scenarios

Author: [Your Group Names]
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

import mibian

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class BlackScholesGreeks:
    """Black-Scholes option pricing and Greeks calculation"""

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
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)
        return self.bs.callPrice if self.option_type == 'call' else self.bs.putPrice

    def delta(self):
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0
        return self.bs.callDelta if self.option_type == 'call' else self.bs.putDelta

    def gamma(self):
        return 0 if self.T <= 0 else self.bs.gamma

    def vega(self):
        return 0 if self.T <= 0 else self.bs.vega


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility from option price"""
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


class HedgingSimulatorWithCostBreakdown:
    """Enhanced simulator that tracks all P&L components separately"""

    def __init__(self, data, strike, expiry_date, option_type='call', risk_free_rate=0.065):
        self.data = data.copy()
        self.strike = strike
        self.expiry_date = pd.to_datetime(expiry_date)
        self.option_type = option_type
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
        """Calculate all Greeks for each day"""
        print(f"Calculating Greeks for {self.option_type.upper()} option...")
        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = calculate_implied_volatility(
                row['Settle Price'], S, self.strike, T, self.r, self.option_type
            )
            self.data.at[idx, 'IV'] = iv

            bs = BlackScholesGreeks(S, self.strike, T, self.r, iv, self.option_type)
            self.data.at[idx, 'Delta'] = bs.delta()
            self.data.at[idx, 'Gamma'] = bs.gamma()
            self.data.at[idx, 'Vega'] = bs.vega()

        print("Done!")
        return self.data

    def simulate_unhedged(self, N=100):
        """Unhedged short option position"""
        results = []
        initial_price = self.data.iloc[0]['Settle Price']

        for idx, row in self.data.iterrows():
            current_price = row['Settle Price']
            pnl_per_contract = initial_price - current_price
            total_pnl = pnl_per_contract * N

            results.append({
                'Date': row['Date'],
                'Option_PnL': total_pnl,
                'Hedge_PnL': 0,
                'Transaction_Costs': 0,
                'Total_PnL': total_pnl
            })

        return pd.DataFrame(results)

    def simulate_delta_hedge_detailed(self, N=100, rebal_threshold=0.10):
        """Delta hedging with detailed cost breakdown"""
        results = []
        initial_option_price = self.data.iloc[0]['Settle Price']
        stock_shares = 0
        cumulative_option_pnl = 0
        cumulative_hedge_pnl = 0
        cumulative_transaction_costs = 0
        last_rebal_delta = 0

        transaction_cost = 0.0001

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            delta = row['Delta']
            current_option_price = row['Settle Price']

            # P&L from option position (we're short)
            option_pnl_change = 0
            if idx > 0:
                prev_option_price = self.data.iloc[idx - 1]['Settle Price']
                option_pnl_change = (prev_option_price - current_option_price) * N
                cumulative_option_pnl += option_pnl_change

            # P&L from stock hedge position
            hedge_pnl_change = 0
            if idx > 0:
                prev_S = self.data.iloc[idx - 1]['Underlying Value']
                hedge_pnl_change = (S - prev_S) * stock_shares
                cumulative_hedge_pnl += hedge_pnl_change

            # Rebalancing
            rebalanced = False
            rebal_cost = 0
            if idx == 0 or abs(delta - last_rebal_delta) > rebal_threshold:
                target_shares = delta * N
                shares_to_trade = abs(target_shares - stock_shares)
                rebal_cost = shares_to_trade * S * transaction_cost
                cumulative_transaction_costs += rebal_cost
                stock_shares = target_shares
                last_rebal_delta = delta
                rebalanced = True

            results.append({
                'Date': row['Date'],
                'Option_PnL': cumulative_option_pnl,
                'Hedge_PnL': cumulative_hedge_pnl,
                'Transaction_Costs': -cumulative_transaction_costs,
                'Total_PnL': cumulative_option_pnl + cumulative_hedge_pnl - cumulative_transaction_costs,
                'Stock_Shares': stock_shares,
                'Rebalanced': rebalanced
            })

        return pd.DataFrame(results)

    def simulate_delta_gamma_hedge_detailed(self, N=100, rebal_threshold=0.10):
        """Delta-Gamma hedging with detailed cost breakdown"""
        results = []

        hedge_strike = self.strike + 200 if self.option_type == 'call' else self.strike - 200

        stock_shares = 0
        hedge_contracts = 0

        cumulative_option_pnl = 0
        cumulative_stock_pnl = 0
        cumulative_hedge_option_pnl = 0
        cumulative_transaction_costs = 0

        last_rebal_gamma = 0
        prev_option_price = self.data.iloc[0]['Settle Price']
        prev_hedge_price = 0
        prev_S = self.data.iloc[0]['Underlying Value']

        transaction_cost = 0.0001

        for idx, row in self.data.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            gamma = row['Gamma']
            delta = row['Delta']
            current_option_price = row['Settle Price']

            # Calculate hedge option
            bs_hedge = BlackScholesGreeks(S, hedge_strike, T, self.r, iv, self.option_type)
            current_hedge_price = bs_hedge.price()
            current_hedge_gamma = bs_hedge.gamma()
            current_hedge_delta = bs_hedge.delta()

            # P&L calculations
            if idx > 0:
                option_pnl = (prev_option_price - current_option_price) * N
                hedge_pnl = (current_hedge_price - prev_hedge_price) * hedge_contracts
                stock_pnl = (S - prev_S) * stock_shares

                cumulative_option_pnl += option_pnl
                cumulative_hedge_option_pnl += hedge_pnl
                cumulative_stock_pnl += stock_pnl

            # Rebalancing
            rebalanced = False
            if idx == 0 or abs(gamma - last_rebal_gamma) > rebal_threshold * max(gamma, 0.0001):

                # Gamma hedge
                if current_hedge_gamma > 0.00001:
                    new_hedge_contracts = min((gamma * N * 0.3) / current_hedge_gamma, N * 1.0)
                else:
                    new_hedge_contracts = 0

                contracts_to_trade = abs(new_hedge_contracts - hedge_contracts)
                if contracts_to_trade > 0.01:
                    trade_cost = contracts_to_trade * current_hedge_price * transaction_cost

                    # Cost of buying/selling hedge options
                    if new_hedge_contracts > hedge_contracts:
                        # Buying more hedge options
                        purchase_cost = (new_hedge_contracts - hedge_contracts) * current_hedge_price
                        cumulative_hedge_option_pnl -= purchase_cost
                    else:
                        # Selling hedge options
                        sale_proceeds = (hedge_contracts - new_hedge_contracts) * current_hedge_price
                        cumulative_hedge_option_pnl += sale_proceeds

                    cumulative_transaction_costs += trade_cost
                    hedge_contracts = new_hedge_contracts
                    rebalanced = True

                # Delta hedge
                net_delta = delta * N - current_hedge_delta * hedge_contracts
                new_stock_shares = net_delta
                shares_to_trade = abs(new_stock_shares - stock_shares)

                if shares_to_trade > 0.01:
                    trade_cost = shares_to_trade * S * transaction_cost
                    cumulative_transaction_costs += trade_cost
                    stock_shares = new_stock_shares
                    rebalanced = True

                last_rebal_gamma = gamma

            prev_option_price = current_option_price
            prev_hedge_price = current_hedge_price
            prev_S = S

            results.append({
                'Date': row['Date'],
                'Option_PnL': cumulative_option_pnl,
                'Hedge_PnL': cumulative_stock_pnl + cumulative_hedge_option_pnl,
                'Transaction_Costs': -cumulative_transaction_costs,
                'Total_PnL': cumulative_option_pnl + cumulative_stock_pnl + cumulative_hedge_option_pnl - cumulative_transaction_costs,
                'Hedge_Contracts': hedge_contracts,
                'Stock_Shares': stock_shares,
                'Rebalanced': rebalanced
            })

        return pd.DataFrame(results)


def create_cost_breakdown_visualization(unhedged, delta, dg, option_type, output_dir):
    """Create visualization showing cost breakdown of hedging"""

    fig = plt.figure(figsize=(20, 12))
    opt_label = option_type.upper()

    # 1. Stacked P&L Components - Delta Hedge
    ax1 = plt.subplot(2, 3, 1)
    dates = delta['Date']

    ax1.fill_between(dates, 0, delta['Option_PnL'], label='Option P&L', alpha=0.7, color='#3498DB')
    ax1.fill_between(dates, delta['Option_PnL'], delta['Option_PnL'] + delta['Hedge_PnL'],
                     label='Hedge P&L', alpha=0.7, color='#2ECC71')
    ax1.fill_between(dates, delta['Option_PnL'] + delta['Hedge_PnL'], delta['Total_PnL'],
                     label='Transaction Costs', alpha=0.7, color='#E74C3C')
    ax1.plot(dates, delta['Total_PnL'], 'k-', linewidth=2, label='Net P&L')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f'{opt_label} - Delta Hedge: P&L Breakdown', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Cumulative P&L (₹)', fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Stacked P&L Components - Delta-Gamma Hedge
    ax2 = plt.subplot(2, 3, 2)

    ax2.fill_between(dates, 0, dg['Option_PnL'], label='Option P&L', alpha=0.7, color='#3498DB')
    ax2.fill_between(dates, dg['Option_PnL'], dg['Option_PnL'] + dg['Hedge_PnL'],
                     label='Hedge P&L', alpha=0.7, color='#2ECC71')
    ax2.fill_between(dates, dg['Option_PnL'] + dg['Hedge_PnL'], dg['Total_PnL'],
                     label='Transaction Costs', alpha=0.7, color='#E74C3C')
    ax2.plot(dates, dg['Total_PnL'], 'k-', linewidth=2, label='Net P&L')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title(f'{opt_label} - Delta-Gamma: P&L Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Cumulative P&L (₹)', fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Final P&L Component Breakdown - Bar Chart
    ax3 = plt.subplot(2, 3, 3)

    strategies = ['Unhedged', 'Delta\nHedged', 'Delta-Gamma\nHedged']

    option_pnl_final = [
        unhedged['Option_PnL'].iloc[-1],
        delta['Option_PnL'].iloc[-1],
        dg['Option_PnL'].iloc[-1]
    ]

    hedge_pnl_final = [
        0,
        delta['Hedge_PnL'].iloc[-1],
        dg['Hedge_PnL'].iloc[-1]
    ]

    costs_final = [
        0,
        delta['Transaction_Costs'].iloc[-1],
        dg['Transaction_Costs'].iloc[-1]
    ]

    x = np.arange(len(strategies))
    width = 0.6

    p1 = ax3.bar(x, option_pnl_final, width, label='Option P&L', color='#3498DB', alpha=0.8)
    p2 = ax3.bar(x, hedge_pnl_final, width, bottom=option_pnl_final,
                 label='Hedge P&L', color='#2ECC71', alpha=0.8)
    p3 = ax3.bar(x, costs_final, width, bottom=np.array(option_pnl_final) + np.array(hedge_pnl_final),
                 label='Transaction Costs', color='#E74C3C', alpha=0.8)

    ax3.set_ylabel('Final P&L (₹)', fontweight='bold')
    ax3.set_title(f'{opt_label} - Final P&L Components', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies)
    ax3.legend(fontsize=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add total values on top
    for i, (opt, hdg, cost) in enumerate(zip(option_pnl_final, hedge_pnl_final, costs_final)):
        total = opt + hdg + cost
        ax3.text(i, total, f'₹{total:,.0f}', ha='center', va='bottom' if total > 0 else 'top',
                 fontweight='bold', fontsize=10)

    # 4. Hedging Cost vs Benefit Comparison - Delta
    ax4 = plt.subplot(2, 3, 4)

    # Calculate "what if unhedged" vs "actual hedged"
    dates_delta = delta['Date']
    unhedged_trajectory = unhedged['Total_PnL']
    hedged_trajectory = delta['Total_PnL']

    ax4.plot(dates, unhedged_trajectory, 'r-', linewidth=2.5, label='Unhedged (Actual)', alpha=0.7)
    ax4.plot(dates_delta, hedged_trajectory, 'b-', linewidth=2.5, label='Delta Hedged (Actual)')
    ax4.fill_between(dates_delta, unhedged_trajectory, hedged_trajectory,
                     where=(hedged_trajectory < unhedged_trajectory),
                     interpolate=True, alpha=0.3, color='red', label='Cost of Hedging')
    ax4.fill_between(dates_delta, unhedged_trajectory, hedged_trajectory,
                     where=(hedged_trajectory > unhedged_trajectory),
                     interpolate=True, alpha=0.3, color='green', label='Benefit of Hedging')

    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title(f'{opt_label} - Cost vs Benefit (Delta Hedge)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('P&L (₹)', fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # 5. Cumulative Transaction Costs Over Time
    ax5 = plt.subplot(2, 3, 5)

    ax5.plot(dates, -delta['Transaction_Costs'], linewidth=2.5, label='Delta Hedge', color='#3498DB')
    ax5.plot(dates, -dg['Transaction_Costs'], linewidth=2.5, label='Delta-Gamma', color='#2ECC71')
    ax5.set_title('Cumulative Transaction Costs', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontweight='bold')
    ax5.set_ylabel('Cumulative Costs (₹)', fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45)

    # Annotate final costs
    final_delta_cost = -delta['Transaction_Costs'].iloc[-1]
    final_dg_cost = -dg['Transaction_Costs'].iloc[-1]
    ax5.text(0.02, 0.98, f'Delta: ₹{final_delta_cost:,.0f}\nDelta-Gamma: ₹{final_dg_cost:,.0f}',
             transform=ax5.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. Key Metrics Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = [
        ['Metric', 'Unhedged', 'Delta', 'D-Gamma'],
        ['Option P&L', f'₹{unhedged["Option_PnL"].iloc[-1]:,.0f}',
         f'₹{delta["Option_PnL"].iloc[-1]:,.0f}',
         f'₹{dg["Option_PnL"].iloc[-1]:,.0f}'],
        ['Hedge P&L', '₹0',
         f'₹{delta["Hedge_PnL"].iloc[-1]:,.0f}',
         f'₹{dg["Hedge_PnL"].iloc[-1]:,.0f}'],
        ['Transaction Costs', '₹0',
         f'₹{delta["Transaction_Costs"].iloc[-1]:,.0f}',
         f'₹{dg["Transaction_Costs"].iloc[-1]:,.0f}'],
        ['Net P&L', f'₹{unhedged["Total_PnL"].iloc[-1]:,.0f}',
         f'₹{delta["Total_PnL"].iloc[-1]:,.0f}',
         f'₹{dg["Total_PnL"].iloc[-1]:,.0f}'],
        ['Risk (Std Dev)', f'₹{unhedged["Total_PnL"].std():,.0f}',
         f'₹{delta["Total_PnL"].std():,.0f}',
         f'₹{dg["Total_PnL"].std():,.0f}'],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#D5E8F0')
        table[(0, i)].set_text_props(weight='bold')

    ax6.set_title(f'{opt_label} - Summary Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_cost_breakdown_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {filename}")


def main():
    """Main analysis function with cost breakdown"""

    print("=" * 80)
    print("HEDGING COST BREAKDOWN ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = 'Report_Outputs_COST_BREAKDOWN'
    os.makedirs(output_dir, exist_ok=True)

    # Analysis parameters
    call_file = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    put_file = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'

    contracts = 100

    # Process CALL option
    print("\nANALYZING CALL OPTION WITH COST BREAKDOWN")
    print("=" * 80)

    df_call = pd.read_csv(call_file)
    df_call.columns = df_call.columns.str.strip()
    df_call = df_call[['Date', 'Settle Price', 'Underlying Value', 'Strike Price', 'Expiry']].copy()

    sim_call = HedgingSimulatorWithCostBreakdown(df_call, df_call['Strike Price'].iloc[0],
                                                 df_call['Expiry'].iloc[0], 'call')
    data_call = sim_call.calculate_greeks()

    print("\nRunning simulations with cost tracking...")
    call_unhedged = sim_call.simulate_unhedged(contracts)
    call_delta = sim_call.simulate_delta_hedge_detailed(contracts)
    call_dg = sim_call.simulate_delta_gamma_hedge_detailed(contracts)

    create_cost_breakdown_visualization(call_unhedged, call_delta, call_dg, 'call', output_dir)

    # Process PUT option
    print("\nANALYZING PUT OPTION WITH COST BREAKDOWN")
    print("=" * 80)

    df_put = pd.read_csv(put_file)
    df_put.columns = df_put.columns.str.strip()
    df_put = df_put[['Date', 'Settle Price', 'Underlying Value', 'Strike Price', 'Expiry']].copy()

    sim_put = HedgingSimulatorWithCostBreakdown(df_put, df_put['Strike Price'].iloc[0],
                                                df_put['Expiry'].iloc[0], 'put')
    data_put = sim_put.calculate_greeks()

    print("\nRunning simulations with cost tracking...")
    put_unhedged = sim_put.simulate_unhedged(contracts)
    put_delta = sim_put.simulate_delta_hedge_detailed(contracts)
    put_dg = sim_put.simulate_delta_gamma_hedge_detailed(contracts)

    create_cost_breakdown_visualization(put_unhedged, put_delta, put_dg, 'put', output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("CALL OPTION COST BREAKDOWN:")
    print("=" * 80)
    print(f"Unhedged Final P&L:      ₹{call_unhedged['Total_PnL'].iloc[-1]:,.2f}")
    print(f"Delta Hedged:")
    print(f"  - Option P&L:          ₹{call_delta['Option_PnL'].iloc[-1]:,.2f}")
    print(f"  - Hedge P&L:           ₹{call_delta['Hedge_PnL'].iloc[-1]:,.2f}")
    print(f"  - Transaction Costs:   ₹{call_delta['Transaction_Costs'].iloc[-1]:,.2f}")
    print(f"  - TOTAL:               ₹{call_delta['Total_PnL'].iloc[-1]:,.2f}")
    print(f"\nDelta-Gamma Hedged:")
    print(f"  - Option P&L:          ₹{call_dg['Option_PnL'].iloc[-1]:,.2f}")
    print(f"  - Hedge P&L:           ₹{call_dg['Hedge_PnL'].iloc[-1]:,.2f}")
    print(f"  - Transaction Costs:   ₹{call_dg['Transaction_Costs'].iloc[-1]:,.2f}")
    print(f"  - TOTAL:               ₹{call_dg['Total_PnL'].iloc[-1]:,.2f}")

    print("\n" + "=" * 80)
    print("PUT OPTION COST BREAKDOWN:")
    print("=" * 80)
    print(f"Unhedged Final P&L:      ₹{put_unhedged['Total_PnL'].iloc[-1]:,.2f}")
    print(f"Delta Hedged:")
    print(f"  - Option P&L:          ₹{put_delta['Option_PnL'].iloc[-1]:,.2f}")
    print(f"  - Hedge P&L:           ₹{put_delta['Hedge_PnL'].iloc[-1]:,.2f}")
    print(f"  - Transaction Costs:   ₹{put_delta['Transaction_Costs'].iloc[-1]:,.2f}")
    print(f"  - TOTAL:               ₹{put_delta['Total_PnL'].iloc[-1]:,.2f}")
    print(f"\nDelta-Gamma Hedged:")
    print(f"  - Option P&L:          ₹{put_dg['Option_PnL'].iloc[-1]:,.2f}")
    print(f"  - Hedge P&L:           ₹{put_dg['Hedge_PnL'].iloc[-1]:,.2f}")
    print(f"  - Transaction Costs:   ₹{put_dg['Transaction_Costs'].iloc[-1]:,.2f}")
    print(f"  - TOTAL:               ₹{put_dg['Total_PnL'].iloc[-1]:,.2f}")

    print("\n" + "=" * 80)
    print("✅ COST BREAKDOWN ANALYSIS COMPLETE!")
    print(f"✅ Charts saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()