"""
OPTIONS HEDGING SIMULATION - EDUCATIONAL VERSION
Strategies: Unhedged, Delta, Delta-Gamma, Delta-Vega

FEATURES:
1. Fixed Delta hedge with rebalancing threshold
2. Delta-Gamma hedge (70% partial gamma hedging)
3. Delta-Vega hedge (volatility risk management)
4. Zero transaction costs (educational simplification)

Author: Enhanced Implementation
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
    """Black-Scholes Greeks calculator with Vega"""

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

    def vega(self):
        return self.bs.vega


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
    ivs, deltas, gammas, vegas = [], [], [], []
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
        vegas.append(bs.vega())

    df['IV'] = ivs
    df['Delta'] = deltas
    df['Gamma'] = gammas
    df['Vega'] = vegas

    # Calculate Greeks for HEDGE option
    print("Calculating Greeks for hedge option...")
    h_deltas, h_gammas, h_vegas = [], [], []
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
        h_vegas.append(bs_h.vega())

    df['Hedge_Delta'] = h_deltas
    df['Hedge_Gamma'] = h_gammas
    df['Hedge_Vega'] = h_vegas

    print("✓ Data preparation complete")

    return df


# =============================================================================
# STRATEGY 1: UNHEDGED
# =============================================================================

def simulate_unhedged(df, N=100):
    """Strategy 1: Short N options, no hedge"""
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
# STRATEGY 2: DELTA HEDGE (FIXED with rebalancing threshold)
# =============================================================================

def simulate_delta_hedge(df, N=100, rebalance_threshold=0.10):
    """
    Strategy 2: Short N options + Delta hedge with stock

    FIXED: Now uses rebalancing threshold to avoid over-trading
    Only rebalances when delta exposure changes by more than threshold

    Args:
        rebalance_threshold: Minimum change in delta exposure to trigger rebalancing
                            0.10 means rebalance if delta changes by more than 10%
    """
    print("\n--- Running DELTA HEDGE simulation ---")
    print(f"Rebalancing threshold: {rebalance_threshold * 100:.1f}%")

    results = []

    # Initial setup
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    last_target_stock = 0  # Track last target to calculate change

    print(f"Initial premium: ₹{cash:,.2f}")

    for i, row in df.iterrows():
        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N

        # Calculate delta exposure
        short_delta = -1 * row['Delta'] * N
        target_stock = -short_delta

        # FIXED: Check if rebalancing is needed
        if i == 0:
            # Always trade on first day to establish hedge
            rebalance_needed = True
        else:
            # Calculate percentage change in target position
            if last_target_stock != 0:
                pct_change = abs((target_stock - last_target_stock) / last_target_stock)
            else:
                pct_change = abs(target_stock) if target_stock != 0 else 0

            rebalance_needed = pct_change > rebalance_threshold

        # Execute stock trade only if rebalancing threshold exceeded
        if rebalance_needed:
            stock_trade = target_stock - stock_pos
            stock_trade_cost = stock_trade * S
            cash -= stock_trade_cost
            stock_pos = target_stock
            last_target_stock = target_stock

        # Mark to market
        stock_value = stock_pos * S
        total_pnl = cash + stock_value - main_opt_value

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': 0
        })

    df_result = pd.DataFrame(results)

    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    print(f"Risk (Std Dev): ₹{df_result['PnL'].std():,.2f}")

    return df_result


# =============================================================================
# STRATEGY 3: DELTA-GAMMA HEDGE
# =============================================================================

def simulate_delta_gamma_hedge(df, N=100, gamma_hedge_ratio=0.70):
    """
    Strategy 3: Delta-Gamma Hedge

    Uses partial gamma hedging (70% instead of 100%)
    This reduces hedge costs while maintaining most of the protection

    Args:
        gamma_hedge_ratio: Percentage of gamma to hedge (0.7 = 70%)
    """
    print("\n--- Running DELTA-GAMMA HEDGE simulation ---")
    print(f"Gamma hedge ratio: {gamma_hedge_ratio * 100:.0f}%")

    results = []

    # Initial setup
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    hedge_opt_pos = 0

    print(f"Initial premium: ₹{cash:,.2f}")

    for i, row in df.iterrows():
        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N
        hedge_opt_price = row['Hedge_Price']

        # STEP 1: PARTIAL GAMMA NEUTRALIZATION (70% hedge)
        short_gamma = -1 * row['Gamma'] * N
        hedge_gamma_per_contract = row['Hedge_Gamma']

        if hedge_gamma_per_contract > 1e-7:
            # Only hedge 70% of gamma exposure
            target_hedge_opts = gamma_hedge_ratio * (-short_gamma / hedge_gamma_per_contract)
            target_hedge_opts = np.clip(target_hedge_opts, 0, N * 2)  # Cap at 2x
        else:
            target_hedge_opts = 0

        # Execute hedge option trade
        hedge_trade = target_hedge_opts - hedge_opt_pos
        hedge_trade_cost = hedge_trade * hedge_opt_price
        cash -= hedge_trade_cost
        hedge_opt_pos = target_hedge_opts

        # STEP 2: DELTA NEUTRALIZATION
        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_opt_pos * row['Hedge_Delta']
        current_net_delta = short_delta + hedge_delta

        target_stock = -current_net_delta

        # Execute stock trade
        stock_trade = target_stock - stock_pos
        stock_trade_cost = stock_trade * S
        cash -= stock_trade_cost
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

    return df_result


# =============================================================================
# STRATEGY 4: DELTA-VEGA HEDGE
# =============================================================================

def simulate_vega_hedge(df, N=100):
    """
    Strategy 4: Delta + Vega Hedge

    Hedges both directional risk (delta) and volatility risk (vega)
    Uses hedge options for vega neutralization, stock for residual delta
    """
    print("\n--- Running DELTA-VEGA HEDGE simulation ---")

    results = []

    # Initial setup
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    hedge_opt_pos = 0

    print(f"Initial premium: ₹{cash:,.2f}")

    for i, row in df.iterrows():
        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N
        hedge_opt_price = row['Hedge_Price']

        # STEP 1: VEGA NEUTRALIZATION
        short_vega = -1 * row['Vega'] * N
        hedge_vega_per_contract = row['Hedge_Vega']

        if hedge_vega_per_contract > 1e-7:
            target_hedge_opts = -short_vega / hedge_vega_per_contract
            target_hedge_opts = np.clip(target_hedge_opts, 0, N * 2)
        else:
            target_hedge_opts = 0

        # Execute hedge option trade
        hedge_trade = target_hedge_opts - hedge_opt_pos
        hedge_trade_cost = hedge_trade * hedge_opt_price
        cash -= hedge_trade_cost
        hedge_opt_pos = target_hedge_opts

        # STEP 2: DELTA NEUTRALIZATION (with stock)
        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_opt_pos * row['Hedge_Delta']
        current_net_delta = short_delta + hedge_delta

        target_stock = -current_net_delta

        # Execute stock trade
        stock_trade = target_stock - stock_pos
        stock_trade_cost = stock_trade * S
        cash -= stock_trade_cost
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

    return df_result


# =============================================================================
# ENHANCED VISUALIZATION
# =============================================================================

def create_comprehensive_comparison(data, strategies_dict, option_type, output_dir):
    """
    Create comprehensive comparison of hedging strategies

    strategies_dict = {
        'Unhedged': df,
        'Delta': df,
        'Delta-Gamma': df,
        'Delta-Vega': df
    }
    """

    fig = plt.figure(figsize=(20, 14))
    opt_label = option_type.upper()

    colors = {
        'Unhedged': 'red',
        'Delta': 'blue',
        'Delta-Gamma': 'green',
        'Delta-Vega': 'purple'
    }

    # 1. P&L Comparison (All Strategies)
    ax1 = plt.subplot(3, 3, 1)
    for name, df in strategies_dict.items():
        ax1.plot(df['Date'], df['PnL'], label=name, linewidth=2.5,
                 color=colors.get(name, 'gray'), alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f'{opt_label}: P&L Evolution Comparison',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('P&L (₹)', fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Risk Comparison (Std Dev)
    ax2 = plt.subplot(3, 3, 2)
    names = list(strategies_dict.keys())
    risks = [df['PnL'].std() for df in strategies_dict.values()]
    bars = ax2.bar(range(len(names)), risks,
                   color=[colors.get(n, 'gray') for n in names],
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_title('Risk Comparison\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Std Dev (₹)', fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, risks):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'₹{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 3. Final P&L
    ax3 = plt.subplot(3, 3, 3)
    final_pnls = [df['PnL'].iloc[-1] for df in strategies_dict.values()]
    colors_pnl = ['green' if x >= 0 else 'red' for x in final_pnls]
    bars = ax3.bar(range(len(names)), final_pnls, color=colors_pnl, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax3.set_title('Final P&L', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Final P&L (₹)', fontweight='bold')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, final_pnls):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'₹{val:,.0f}', ha='center',
                 va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)

    # 4. Stock Positions Over Time
    ax4 = plt.subplot(3, 3, 4)
    for name, df in strategies_dict.items():
        if name != 'Unhedged':  # Unhedged has no stock position
            ax4.plot(df['Date'], df['Stock_Pos'], label=name, linewidth=2,
                     color=colors.get(name, 'gray'), alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Stock Positions Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontweight='bold')
    ax4.set_ylabel('Stock Shares', fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Hedge Option Contracts
    ax5 = plt.subplot(3, 3, 5)
    for name, df in strategies_dict.items():
        if 'Gamma' in name or 'Vega' in name:
            ax5.plot(df['Date'], df['Hedge_Opt_Pos'], label=name, linewidth=2,
                     color=colors.get(name, 'gray'), alpha=0.7)
    ax5.set_title('Hedge Option Contracts\n(Gamma/Vega Strategies)',
                  fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontweight='bold')
    ax5.set_ylabel('Hedge Contracts', fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Max Drawdown Comparison
    ax6 = plt.subplot(3, 3, 6)
    max_pnls = [df['PnL'].max() for df in strategies_dict.values()]
    min_pnls = [df['PnL'].min() for df in strategies_dict.values()]
    drawdowns = [max_val - min_val for max_val, min_val in zip(max_pnls, min_pnls)]

    bars = ax6.bar(range(len(names)), drawdowns,
                   color=[colors.get(n, 'gray') for n in names],
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_title('Maximum Drawdown\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Drawdown (₹)', fontweight='bold')
    ax6.set_xticks(range(len(names)))
    ax6.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, drawdowns):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'₹{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 7. Summary Statistics Table
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')

    # Calculate risk-adjusted returns
    risk_adj_returns = []
    for df in strategies_dict.values():
        final = df['PnL'].iloc[-1]
        std = df['PnL'].std()
        risk_adj_returns.append(final / std if std > 0 else 0)

    # Calculate max loss
    max_losses = [df['PnL'].min() for df in strategies_dict.values()]

    table_data = [
        ['Strategy', 'Final P&L', 'Std Dev', 'Max Loss', 'Risk-Adj Return']
    ]

    for i, name in enumerate(names):
        table_data.append([
            name,
            f'₹{final_pnls[i]:,.0f}',
            f'₹{risks[i]:,.0f}',
            f'₹{max_losses[i]:,.0f}',
            f'{risk_adj_returns[i]:.2f}'
        ])

    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.20, 0.18, 0.18, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#D5E8F0')
        table[(0, i)].set_text_props(weight='bold')

    ax7.set_title(f'{opt_label} - Performance Summary', fontsize=14, fontweight='bold', pad=20)

    # 8. Hedging Effectiveness (Risk Reduction %)
    ax8 = plt.subplot(3, 3, 8)
    baseline_risk = risks[0]  # Unhedged risk
    risk_reductions = [(1 - r / baseline_risk) * 100 if baseline_risk > 0 else 0 for r in risks]

    bars = ax8.bar(range(len(names)), risk_reductions,
                   color=[colors.get(n, 'gray') for n in names],
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax8.set_title('Risk Reduction vs Unhedged\n(Higher = Better)',
                  fontsize=14, fontweight='bold')
    ax8.set_ylabel('Risk Reduction (%)', fontweight='bold')
    ax8.set_xticks(range(len(names)))
    ax8.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, risk_reductions):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 9. Profit Retention vs Unhedged
    ax9 = plt.subplot(3, 3, 9)
    baseline_profit = final_pnls[0]  # Unhedged profit

    if baseline_profit != 0:
        profit_retentions = [(p / baseline_profit) * 100 for p in final_pnls]
    else:
        profit_retentions = [0] * len(final_pnls)

    colors_ret = ['green' if x >= 0 else 'red' for x in profit_retentions]
    bars = ax9.bar(range(len(names)), profit_retentions, color=colors_ret,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax9.set_title('P&L Retention vs Unhedged', fontsize=14, fontweight='bold')
    ax9.set_ylabel('P&L Retention (%)', fontweight='bold')
    ax9.set_xticks(range(len(names)))
    ax9.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax9.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax9.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, profit_retentions):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.1f}%', ha='center',
                 va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_comprehensive_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Chart saved: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function
    """

    # =========================================================================
    # FILE CONFIGURATION
    # =========================================================================

    # CALL option files
    CALL_MAIN_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    CALL_HEDGE_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026 (2).csv'

    # PUT option files
    PUT_MAIN_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'
    PUT_HEDGE_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026 (2).csv'

    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_CONTRACTS = 100

    print("\n" + "=" * 70)
    print("OPTIONS HEDGING ANALYSIS - EDUCATIONAL VERSION")
    print("Zero Transaction Costs | Fixed Delta Rebalancing")
    print("=" * 70)
    print(f"\nContracts: {N_CONTRACTS}")
    print(f"Strategies: 4 (Unhedged + 3 Hedges)")
    print("=" * 70)

    # =========================================================================
    # CALL OPTION ANALYSIS
    # =========================================================================

    if os.path.exists(CALL_MAIN_FILE) and os.path.exists(CALL_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("CALL OPTION ANALYSIS")
        print("=" * 70)

        df_call = load_and_prepare_data(CALL_MAIN_FILE, CALL_HEDGE_FILE, 'call')

        # Run all strategies
        call_strategies = {
            'Unhedged': simulate_unhedged(df_call, N_CONTRACTS),
            'Delta': simulate_delta_hedge(df_call, N_CONTRACTS, rebalance_threshold=0.10),
            'Delta-Gamma': simulate_delta_gamma_hedge(df_call, N_CONTRACTS, gamma_hedge_ratio=0.70),
            'Delta-Vega': simulate_vega_hedge(df_call, N_CONTRACTS)
            }

        # Create comprehensive visualization
        create_comprehensive_comparison(df_call, call_strategies, 'call', OUTPUT_DIR)

        # Save results
        for name, df in call_strategies.items():
            safe_name = name.replace(' ', '_').replace('-', '_')
            df.to_csv(f'{OUTPUT_DIR}/call_{safe_name.lower()}_results.csv', index=False)

    else:
        print("\n⚠️  CALL files not found. Skipping CALL analysis.")

    # =========================================================================
    # PUT OPTION ANALYSIS
    # =========================================================================

    if os.path.exists(PUT_MAIN_FILE) and os.path.exists(PUT_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("PUT OPTION ANALYSIS")
        print("=" * 70)

        df_put = load_and_prepare_data(PUT_MAIN_FILE, PUT_HEDGE_FILE, 'put')

        # Run all strategies
        put_strategies = {
            'Unhedged': simulate_unhedged(df_put, N_CONTRACTS),
            'Delta': simulate_delta_hedge(df_put, N_CONTRACTS, rebalance_threshold=0.10),
            'Delta-Gamma': simulate_delta_gamma_hedge(df_put, N_CONTRACTS, gamma_hedge_ratio=0.70),
            'Delta-Vega': simulate_vega_hedge(df_put, N_CONTRACTS)
        }

        # Create comprehensive visualization
        create_comprehensive_comparison(df_put, put_strategies, 'put', OUTPUT_DIR)

        # Save results
        for name, df in put_strategies.items():
            safe_name = name.replace(' ', '_').replace('-', '_')
            df.to_csv(f'{OUTPUT_DIR}/put_{safe_name.lower()}_results.csv', index=False)

    else:
        print("\n⚠️  PUT files not found. Skipping PUT analysis.")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  📊 call_comprehensive_comparison.png")
    print("  📊 put_comprehensive_comparison.png")
    print("  📄 CSV files for each strategy")
    print("\n" + "=" * 70)
    print("KEY FEATURES:")
    print("=" * 70)
    print("✓ Fixed Delta hedge with 10% rebalancing threshold")
    print("✓ Delta-Gamma hedge with 70% partial gamma hedging")
    print("✓ Delta-Vega hedge for volatility risk management")
    print("✓ Zero transaction costs (educational simplification)")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    main()