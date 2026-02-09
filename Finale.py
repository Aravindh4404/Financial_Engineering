"""
FINANCIAL ENGINEERING: COMPLETE HEDGING SIMULATION
DATA 609 - Final Correct Implementation

This script simulates THREE hedging strategies:
1. Unhedged (short options only)
2. Delta Hedged (short options + stock)
3. Delta-Gamma Hedged (short options + stock + hedge options - USING REAL PRICES)

Author: Final Implementation
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mibian
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# HELPER CLASSES
# =============================================================================

class BlackScholes:
    """Black-Scholes Greeks calculator"""

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = float(S)
        self.K = float(K)
        self.T = max(float(T), 0.0001)
        self.r = float(r)
        self.sigma = max(float(sigma), 0.001)
        self.type = option_type.lower()

        self.bs = mibian.BS(
            [self.S, self.K, self.r * 100, self.T * 365],
            volatility=self.sigma * 100
        )

    def delta(self):
        return self.bs.callDelta if self.type == 'call' else self.bs.putDelta

    def gamma(self):
        return self.bs.gamma


def calculate_implied_vol(price, S, K, T, r, option_type, fallback_iv=0.20):
    """Calculate IV from option price"""
    if T <= 0.001 or price < 0.05:
        return fallback_iv

    try:
        bs = mibian.BS(
            [S, K, r * 100, T * 365],
            callPrice=price if option_type == 'call' else None,
            putPrice=price if option_type == 'put' else None
        )
        return max(0.01, bs.impliedVolatility / 100)
    except:
        return fallback_iv


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(main_file, hedge_file, option_type, r=0.065):
    """
    Load main option and hedge option data, calculate Greeks for both

    Args:
        main_file: CSV for the option we're shorting
        hedge_file: CSV for the option we're buying (for gamma hedge)
        option_type: 'call' or 'put'
        r: risk-free rate

    Returns:
        DataFrame with both options' prices and Greeks
    """
    print(f"\n{'=' * 70}")
    print(f"Loading {option_type.upper()} data...")
    print(f"{'=' * 70}")

    # Load main option data
    df_main = pd.read_csv(main_file)
    df_main.columns = df_main.columns.str.strip()
    df_main['Date'] = pd.to_datetime(df_main['Date'], format='%d-%b-%Y')

    # Load hedge option data
    df_hedge = pd.read_csv(hedge_file)
    df_hedge.columns = df_hedge.columns.str.strip()
    df_hedge['Date'] = pd.to_datetime(df_hedge['Date'], format='%d-%b-%Y')

    # Merge on date
    df_hedge = df_hedge[['Date', 'Settle Price']].rename(
        columns={'Settle Price': 'Hedge_Price'}
    )
    df = pd.merge(df_main, df_hedge, on='Date', how='inner')
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Main strike: {df['Strike Price'].iloc[0]}")
    print(f"Loaded {len(df)} trading days")

    # Get parameters
    strike_main = df['Strike Price'].iloc[0]
    strike_hedge = df['Strike Price'].iloc[0] + 200 if option_type == 'call' else df['Strike Price'].iloc[0] - 200
    expiry = pd.to_datetime(df['Expiry'].iloc[0])

    print(f"Hedge strike: {strike_hedge}")
    print(f"Expiry: {expiry.strftime('%d-%b-%Y')}")

    # Calculate time to expiry
    df['T'] = (expiry - df['Date']).dt.days / 365.0
    df['T'] = df['T'].clip(lower=0.0001)

    # Calculate Greeks for MAIN option
    print("Calculating Greeks for main option...")
    ivs, deltas, gammas = [], [], []
    last_iv = 0.20

    for i, row in df.iterrows():
        iv = calculate_implied_vol(
            row['Settle Price'], row['Underlying Value'],
            strike_main, row['T'], r, option_type, last_iv
        )
        last_iv = iv

        bs = BlackScholes(row['Underlying Value'], strike_main, row['T'], r, iv, option_type)
        ivs.append(iv)
        deltas.append(bs.delta())
        gammas.append(bs.gamma())

    df['IV'] = ivs
    df['Delta'] = deltas
    df['Gamma'] = gammas

    # Calculate Greeks for HEDGE option (using real prices)
    print("Calculating Greeks for hedge option...")
    h_deltas, h_gammas = [], []
    last_h_iv = 0.20

    for i, row in df.iterrows():
        h_iv = calculate_implied_vol(
            row['Hedge_Price'], row['Underlying Value'],
            strike_hedge, row['T'], r, option_type, last_h_iv
        )
        last_h_iv = h_iv

        bs_h = BlackScholes(row['Underlying Value'], strike_hedge, row['T'], r, h_iv, option_type)
        h_deltas.append(bs_h.delta())
        h_gammas.append(bs_h.gamma())

    df['Hedge_Delta'] = h_deltas
    df['Hedge_Gamma'] = h_gammas

    print("✓ Data preparation complete")

    return df


# =============================================================================
# STRATEGY 1: UNHEDGED
# =============================================================================

def simulate_unhedged(df, N=100):
    """
    Strategy 1: Short N options, no hedge

    P&L = Premium received - Current option value
    """
    print("\n--- Running UNHEDGED simulation ---")

    results = []
    entry_price = df.iloc[0]['Settle Price']
    premium_received = entry_price * N

    print(f"Sold {N} options at ₹{entry_price:.2f}")
    print(f"Premium received: ₹{premium_received:,.2f}")

    for i, row in df.iterrows():
        current_value = row['Settle Price'] * N
        pnl = premium_received - current_value

        results.append({
            'Date': row['Date'],
            'PnL': pnl,
            'Stock_Pos': 0,
            'Hedge_Opt_Pos': 0
        })

    df_result = pd.DataFrame(results)

    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    print(f"Risk (Std Dev): ₹{df_result['PnL'].std():,.2f}")

    return df_result


# =============================================================================
# STRATEGY 2: DELTA HEDGE
# =============================================================================

def simulate_delta_hedge(df, N=100, rebalance_threshold=0.10):
    """
    Strategy 2: Short N options + Delta hedge with stock

    Cash Flow Approach:
    - Start with premium
    - Buy/sell stock to match delta
    - P&L = Cash + Stock Value - Option Liability
    """
    print("\n--- Running DELTA HEDGE simulation ---")

    results = []

    # Initial setup
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    last_rebal_delta = 0

    tx_cost_rate = 0.0001  # 1 basis point
    total_tx_costs = 0

    print(f"Initial premium: ₹{cash:,.2f}")

    for i, row in df.iterrows():
        S = row['Underlying Value']
        delta = row['Delta']
        opt_value = row['Settle Price'] * N

        # Rebalancing decision
        rebalanced = False
        if i == 0 or abs(delta - last_rebal_delta) > rebalance_threshold:
            # Target: hold delta*N shares to offset short option's delta
            target_shares = delta * N
            shares_to_trade = target_shares - stock_pos

            if abs(shares_to_trade) > 0.01:
                # Execute trade
                trade_value = shares_to_trade * S
                trade_cost = abs(trade_value) * tx_cost_rate

                cash -= (trade_value + trade_cost)
                total_tx_costs += trade_cost
                stock_pos = target_shares
                last_rebal_delta = delta
                rebalanced = True

        # Mark to market
        stock_value = stock_pos * S
        total_pnl = cash + stock_value - opt_value

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': 0,
            'Rebalanced': rebalanced
        })

    df_result = pd.DataFrame(results)

    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    print(f"Risk (Std Dev): ₹{df_result['PnL'].std():,.2f}")
    print(f"Total transaction costs: ₹{total_tx_costs:,.2f}")
    print(f"Number of rebalances: {df_result['Rebalanced'].sum()}")

    return df_result


# =============================================================================
# STRATEGY 3: DELTA-GAMMA HEDGE (USING REAL HEDGE OPTION PRICES)
# =============================================================================

def simulate_delta_gamma_hedge(df, N=100):
    """
    Strategy 3: Short N options + Gamma hedge with real options + Delta hedge with stock

    Uses REAL market prices for hedge options (not theoretical)

    Steps:
    1. Neutralize gamma by buying hedge options
    2. Neutralize remaining delta with stock
    """
    print("\n--- Running DELTA-GAMMA HEDGE simulation ---")
    print("Using REAL market prices for hedge options")

    results = []

    # Initial setup
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    hedge_opt_pos = 0

    tx_cost_rate = 0.0001
    total_tx_costs = 0

    print(f"Initial premium: ₹{cash:,.2f}")

    for i, row in df.iterrows():
        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N
        hedge_opt_price = row['Hedge_Price']

        # STEP 1: GAMMA NEUTRALIZATION
        # Short option gamma exposure: -Gamma * N
        # Need to buy hedge options to offset

        short_gamma = -1 * row['Gamma'] * N
        hedge_gamma_per_contract = row['Hedge_Gamma']

        if hedge_gamma_per_contract > 1e-7:
            # Contracts needed = -short_gamma / hedge_gamma
            target_hedge_opts = -short_gamma / hedge_gamma_per_contract
            # Safety cap: don't buy more than 3x notional
            target_hedge_opts = np.clip(target_hedge_opts, 0, N * 3)
        else:
            target_hedge_opts = 0

        # Execute hedge option trade (USING REAL PRICE)
        hedge_trade = target_hedge_opts - hedge_opt_pos
        hedge_trade_cost = hedge_trade * hedge_opt_price
        hedge_tx_cost = abs(hedge_trade_cost) * tx_cost_rate

        cash -= (hedge_trade_cost + hedge_tx_cost)
        total_tx_costs += hedge_tx_cost
        hedge_opt_pos = target_hedge_opts

        # STEP 2: DELTA NEUTRALIZATION
        # Net delta = short option delta + hedge option delta + stock delta
        # Set stock position to make net delta = 0

        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_opt_pos * row['Hedge_Delta']
        current_net_delta = short_delta + hedge_delta

        # Stock position to neutralize
        target_stock = -current_net_delta

        # Execute stock trade
        stock_trade = target_stock - stock_pos
        stock_trade_cost = stock_trade * S
        stock_tx_cost = abs(stock_trade_cost) * tx_cost_rate

        cash -= (stock_trade_cost + stock_tx_cost)
        total_tx_costs += stock_tx_cost
        stock_pos = target_stock

        # STEP 3: MARK TO MARKET
        stock_value = stock_pos * S
        hedge_opt_value = hedge_opt_pos * hedge_opt_price

        total_pnl = cash + stock_value + hedge_opt_value - main_opt_value

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': hedge_opt_pos
        })

    df_result = pd.DataFrame(results)

    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    print(f"Risk (Std Dev): ₹{df_result['PnL'].std():,.2f}")
    print(f"Total transaction costs: ₹{total_tx_costs:,.2f}")

    return df_result


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_analysis(data, unhedged, delta, dg, option_type, output_dir):
    """Create detailed analysis charts"""

    fig = plt.figure(figsize=(20, 12))
    opt_label = option_type.upper()

    # 1. P&L Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(unhedged['Date'], unhedged['PnL'],
             label='Unhedged', linewidth=3, color='red', alpha=0.8)
    ax1.plot(delta['Date'], delta['PnL'],
             label='Delta Hedged', linewidth=3, color='blue', alpha=0.8)
    ax1.plot(dg['Date'], dg['PnL'],
             label='Delta-Gamma Hedged', linewidth=3, color='green', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f'{opt_label}: P&L Comparison (ALL 3 STRATEGIES)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('P&L (₹)', fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Risk Comparison
    ax2 = plt.subplot(2, 3, 2)
    strategies = ['Unhedged', 'Delta\nHedged', 'Delta-Gamma\nHedged']
    risks = [
        unhedged['PnL'].std(),
        delta['PnL'].std(),
        dg['PnL'].std()
    ]
    colors = ['red', 'blue', 'green']
    bars = ax2.bar(strategies, risks, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_title('Risk Comparison\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Std Dev (₹)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, risks):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'₹{val:,.0f}', ha='center', va='bottom', fontweight='bold')

    # 3. Final P&L
    ax3 = plt.subplot(2, 3, 3)
    final_pnls = [
        unhedged['PnL'].iloc[-1],
        delta['PnL'].iloc[-1],
        dg['PnL'].iloc[-1]
    ]
    colors_pnl = ['green' if x >= 0 else 'red' for x in final_pnls]
    bars = ax3.bar(strategies, final_pnls, color=colors_pnl, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax3.set_title('Final P&L', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Final P&L (₹)', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, final_pnls):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'₹{val:,.0f}', ha='center',
                 va='bottom' if val >= 0 else 'top', fontweight='bold')

    # 4. Stock Position Over Time
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(delta['Date'], delta['Stock_Pos'],
             label='Delta Hedge', linewidth=2.5, color='blue', alpha=0.7)
    ax4.plot(dg['Date'], dg['Stock_Pos'],
             label='Delta-Gamma Hedge', linewidth=2.5, color='green', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Stock Positions', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('Stock Shares', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Hedge Option Contracts
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(dg['Date'], dg['Hedge_Opt_Pos'], linewidth=2.5, color='green')
    ax5.set_title('Hedge Option Contracts\n(Delta-Gamma Strategy)',
                  fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontweight='bold')
    ax5.set_ylabel('Hedge Contracts', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    risk_reduction_delta = (1 - risks[1] / risks[0]) * 100
    risk_reduction_dg = (1 - risks[2] / risks[0]) * 100

    table_data = [
        ['Metric', 'Unhedged', 'Delta', 'D-Gamma'],
        ['Final P&L', f'₹{final_pnls[0]:,.0f}', f'₹{final_pnls[1]:,.0f}', f'₹{final_pnls[2]:,.0f}'],
        ['Std Dev', f'₹{risks[0]:,.0f}', f'₹{risks[1]:,.0f}', f'₹{risks[2]:,.0f}'],
        ['Risk Reduction', '0%', f'{risk_reduction_delta:.1f}%', f'{risk_reduction_dg:.1f}%'],
        ['Max Gain', f'₹{unhedged["PnL"].max():,.0f}',
         f'₹{delta["PnL"].max():,.0f}', f'₹{dg["PnL"].max():,.0f}'],
        ['Max Loss', f'₹{unhedged["PnL"].min():,.0f}',
         f'₹{delta["PnL"].min():,.0f}', f'₹{dg["PnL"].min():,.0f}'],
    ]

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    for i in range(4):
        table[(0, i)].set_facecolor('#D5E8F0')
        table[(0, i)].set_text_props(weight='bold')

    ax6.set_title(f'{opt_label} - Summary Metrics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_complete_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Chart saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function

    UPDATE THESE FILE PATHS TO MATCH YOUR DATA FILES:
    """

    # =========================================================================
    # FILE CONFIGURATION - UPDATE THESE PATHS
    # =========================================================================

    # CALL option files
    CALL_MAIN_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    CALL_HEDGE_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026 (2).csv'  # Strike +200

    # PUT option files
    PUT_MAIN_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'
    PUT_HEDGE_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026 (2).csv'  # Strike -200

    # =========================================================================

    OUTPUT_DIR = ('finaleoutputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_CONTRACTS = 100  # Number of contracts to short

    print("\n" + "=" * 70)
    print("FINANCIAL ENGINEERING: COMPLETE HEDGING ANALYSIS")
    print("DATA 609 - Final Implementation")
    print("=" * 70)
    print(f"\nContracts: {N_CONTRACTS}")
    print(f"Strategies: Unhedged, Delta Hedge, Delta-Gamma Hedge")
    print("=" * 70)

    # =========================================================================
    # CALL OPTION ANALYSIS
    # =========================================================================

    if os.path.exists(CALL_MAIN_FILE) and os.path.exists(CALL_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("CALL OPTION ANALYSIS")
        print("=" * 70)

        # Load data
        df_call = load_and_prepare_data(CALL_MAIN_FILE, CALL_HEDGE_FILE, 'call')

        # Run all three strategies
        call_unhedged = simulate_unhedged(df_call, N_CONTRACTS)
        call_delta = simulate_delta_hedge(df_call, N_CONTRACTS)
        call_dg = simulate_delta_gamma_hedge(df_call, N_CONTRACTS)

        # Create visualizations
        create_comprehensive_analysis(df_call, call_unhedged, call_delta,
                                      call_dg, 'call', OUTPUT_DIR)

        # Save results
        call_dg.to_csv(f'{OUTPUT_DIR}/call_delta_gamma_results.csv', index=False)

    else:
        print("\n⚠️  CALL files not found. Skipping CALL analysis.")

    # =========================================================================
    # PUT OPTION ANALYSIS
    # =========================================================================

    if os.path.exists(PUT_MAIN_FILE) and os.path.exists(PUT_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("PUT OPTION ANALYSIS")
        print("=" * 70)

        # Load data
        df_put = load_and_prepare_data(PUT_MAIN_FILE, PUT_HEDGE_FILE, 'put')

        # Run all three strategies
        put_unhedged = simulate_unhedged(df_put, N_CONTRACTS)
        put_delta = simulate_delta_hedge(df_put, N_CONTRACTS)
        put_dg = simulate_delta_gamma_hedge(df_put, N_CONTRACTS)

        # Create visualizations
        create_comprehensive_analysis(df_put, put_unhedged, put_delta,
                                      put_dg, 'put', OUTPUT_DIR)

        # Save results
        put_dg.to_csv(f'{OUTPUT_DIR}/put_delta_gamma_results.csv', index=False)

    else:
        print("\n⚠️  PUT files not found. Skipping PUT analysis.")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("  - call_complete_analysis.png")
    print("  - put_complete_analysis.png")
    print("  - call_delta_gamma_results.csv")
    print("  - put_delta_gamma_results.csv")
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKLIST:")
    print("=" * 70)
    print("✓ Hedging should REDUCE risk (std dev lower for hedged)")
    print("✓ Delta-Gamma should have LOWEST risk")
    print("✓ Final P&L may be slightly lower for hedged (transaction costs)")
    print("✓ NO massive multi-million losses!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()