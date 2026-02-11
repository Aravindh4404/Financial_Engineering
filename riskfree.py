"""
OPTIONS HEDGING SIMULATION - TEACHING VERSION
Simple, clear visualizations for educational purposes

Focuses on:
1. Understanding underlying vs option price relationship
2. Seeing P&L evolution clearly
3. Visualizing when and how hedging happens

Author: Educational Refactor
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mibian
import warnings
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# HELPER CLASSES (UNCHANGED)
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
        # mibian returns positive deltas for both calls and puts
        # But puts should have negative delta by convention
        raw_delta = self.bs.callDelta if self.type == 'call' else self.bs.putDelta

        # FIX: Convert PUT delta to negative if mibian returns positive
        if self.type == 'put' and raw_delta > 0:
            return -raw_delta
        return raw_delta

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
# DATA PREPARATION (UNCHANGED)
# =============================================================================

def load_and_prepare_data(main_file, hedge_file, option_type, r=0.065):
    """Load main option and hedge option data, calculate Greeks for both"""
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
# SIMULATION FUNCTIONS WITH TRADE TRACKING
# =============================================================================

def simulate_unhedged(df, N=100, r=0.065):  # ADD r parameter
    """Strategy 1: Short N options, no hedge"""
    print("\n--- Running UNHEDGED simulation ---")

    results = []
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N  # CHANGE: was premium_received

    for i, row in df.iterrows():
        # ADD THESE 3 LINES - Apply risk-free rate to cash
        if i > 0:
            days = (row['Date'] - df.iloc[i - 1]['Date']).days
            cash *= np.exp(r * days / 365)

        current_value = row['Settle Price'] * N
        pnl = cash - current_value  # CHANGE: was premium_received

        results.append({
            'Date': row['Date'],
            'PnL': pnl,
            'Stock_Pos': 0,
            'Hedge_Opt_Pos': 0,
            'Stock_Trade': 0,
            'Hedge_Trade': 0
        })

    df_result = pd.DataFrame(results)
    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    return df_result


def simulate_delta_hedge(df, N=100, rebalance_threshold=0.20, r=0.065):  # ADD r parameter
    """Strategy 2: Short N options + Delta hedge with stock - TRACKS TRADES"""
    print("\n--- Running DELTA HEDGE simulation ---")
    print(f"Rebalancing threshold: {rebalance_threshold * 100:.1f}%")

    results = []
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    last_target_stock = 0

    for i, row in df.iterrows():
        # ADD THESE 3 LINES - Apply risk-free rate to cash
        if i > 0:
            days = (row['Date'] - df.iloc[i - 1]['Date']).days
            cash *= np.exp(r * days / 365)

        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N

        # Calculate delta exposure
        short_delta = -1 * row['Delta'] * N
        target_stock = -short_delta

        # Check if rebalancing needed
        if i == 0:
            rebalance_needed = True
        else:
            if last_target_stock != 0:
                pct_change = abs((target_stock - last_target_stock) / last_target_stock)
            else:
                pct_change = abs(target_stock) if target_stock != 0 else 0
            rebalance_needed = pct_change > rebalance_threshold

        # Execute stock trade
        stock_trade = 0
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
            'Hedge_Opt_Pos': 0,
            'Stock_Trade': abs(stock_trade),  # Track magnitude of trade
            'Hedge_Trade': 0
        })

    df_result = pd.DataFrame(results)
    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    return df_result


def simulate_delta_gamma_hedge(df, N=100, gamma_hedge_ratio=0.70, r=0.065):  # ADD r parameter
    """Strategy 3: Delta-Gamma Hedge - TRACKS TRADES"""
    print("\n--- Running DELTA-GAMMA HEDGE simulation ---")
    print(f"Gamma hedge ratio: {gamma_hedge_ratio * 100:.0f}%")

    results = []
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    hedge_opt_pos = 0

    for i, row in df.iterrows():
        # ADD THESE 3 LINES - Apply risk-free rate to cash
        if i > 0:
            days = (row['Date'] - df.iloc[i - 1]['Date']).days
            cash *= np.exp(r * days / 365)

        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N
        hedge_opt_price = row['Hedge_Price']

        # STEP 1: Gamma neutralization
        short_gamma = -1 * row['Gamma'] * N
        hedge_gamma_per_contract = row['Hedge_Gamma']

        if hedge_gamma_per_contract > 1e-7:
            target_hedge_opts = gamma_hedge_ratio * (-short_gamma / hedge_gamma_per_contract)
            target_hedge_opts = np.clip(target_hedge_opts, 0, N * 2)
        else:
            target_hedge_opts = 0

        # Execute hedge option trade
        hedge_trade = target_hedge_opts - hedge_opt_pos
        hedge_trade_cost = hedge_trade * hedge_opt_price
        cash -= hedge_trade_cost
        hedge_opt_pos = target_hedge_opts

        # STEP 2: Delta neutralization
        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_opt_pos * row['Hedge_Delta']
        current_net_delta = short_delta + hedge_delta
        target_stock = -current_net_delta

        # Execute stock trade
        stock_trade = target_stock - stock_pos
        stock_trade_cost = stock_trade * S
        cash -= stock_trade_cost
        stock_pos = target_stock

        # Mark to market
        stock_value = stock_pos * S
        hedge_opt_value = hedge_opt_pos * hedge_opt_price
        total_pnl = cash + stock_value + hedge_opt_value - main_opt_value

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': hedge_opt_pos,
            'Stock_Trade': abs(stock_trade),
            'Hedge_Trade': abs(hedge_trade)
        })

    df_result = pd.DataFrame(results)
    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    return df_result


def simulate_vega_hedge(df, N=100, r=0.065):  # ADD r parameter
    """Strategy 4: Delta + Vega Hedge - TRACKS TRADES"""
    print("\n--- Running DELTA-VEGA HEDGE simulation ---")

    results = []
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N
    stock_pos = 0
    hedge_opt_pos = 0

    for i, row in df.iterrows():
        # ADD THESE 3 LINES - Apply risk-free rate to cash
        if i > 0:
            days = (row['Date'] - df.iloc[i - 1]['Date']).days
            cash *= np.exp(r * days / 365)

        S = row['Underlying Value']
        main_opt_value = row['Settle Price'] * N
        hedge_opt_price = row['Hedge_Price']

        # STEP 1: Vega neutralization
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

        # STEP 2: Delta neutralization
        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_opt_pos * row['Hedge_Delta']
        current_net_delta = short_delta + hedge_delta
        target_stock = -current_net_delta

        # Execute stock trade
        stock_trade = target_stock - stock_pos
        stock_trade_cost = stock_trade * S
        cash -= stock_trade_cost
        stock_pos = target_stock

        # Mark to market
        stock_value = stock_pos * S
        hedge_opt_value = hedge_opt_pos * hedge_opt_price
        total_pnl = cash + stock_value + hedge_opt_value - main_opt_value

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': hedge_opt_pos,
            'Stock_Trade': abs(stock_trade),
            'Hedge_Trade': abs(hedge_trade)
        })

    df_result = pd.DataFrame(results)
    print(f"Final P&L: ₹{df_result['PnL'].iloc[-1]:,.2f}")
    return df_result


# =============================================================================
# TEACHING-ORIENTED VISUALIZATIONS
# =============================================================================

def plot_underlying_and_option_prices(data, option_type, output_dir):
    """
    PLOT 1: Show underlying price vs option prices over time

    Primary Y-axis: NIFTY underlying
    Secondary Y-axis: Option prices (main + hedge)
    """
    fig, ax1 = plt.figure(figsize=(14, 7)), plt.gca()

    opt_label = option_type.upper()

    # Primary axis: Underlying
    color_underlying = 'black'
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NIFTY Underlying Level', fontsize=14, fontweight='bold', color=color_underlying)
    ax1.plot(data['Date'], data['Underlying Value'],
             color=color_underlying, linewidth=3, label='NIFTY Underlying', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_underlying, labelsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Option prices
    ax2 = ax1.twinx()
    color_main = 'red' if option_type == 'call' else 'blue'
    color_hedge = 'orange' if option_type == 'call' else 'cyan'

    ax2.set_ylabel('Option Prices (₹)', fontsize=14, fontweight='bold', color=color_main)
    ax2.plot(data['Date'], data['Settle Price'],
             color=color_main, linewidth=2.5, label=f'Short {opt_label} (Strike: {data["Strike Price"].iloc[0]})',
             linestyle='-', marker='o', markersize=3, alpha=0.8)
    ax2.plot(data['Date'], data['Hedge_Price'],
             color=color_hedge, linewidth=2.5, label=f'Hedge {opt_label}',
             linestyle='--', marker='s', markersize=3, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_main, labelsize=12)

    # Title and legends
    plt.title(f'{opt_label} Options: Underlying vs Option Prices Over Time',
              fontsize=16, fontweight='bold', pad=20)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_underlying_vs_options.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_pnl_comparison(strategies_dict, option_type, output_dir):
    """
    PLOT 2: Simple P&L comparison over time

    One line per strategy, clear legend, zero line
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    opt_label = option_type.upper()

    colors = {
        'Unhedged': '#e74c3c',  # Red
        'Delta': '#3498db',  # Blue
        'Delta-Gamma': '#2ecc71',  # Green
        'Delta-Vega': '#9b59b6'  # Purple
    }

    for name, df in strategies_dict.items():
        ax.plot(df['Date'], df['PnL'],
                label=name, linewidth=3, color=colors.get(name, 'gray'), alpha=0.85)

    # Zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')

    # Labels and title
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Profit & Loss (₹)', fontsize=14, fontweight='bold')
    ax.set_title(f'{opt_label} Options: Strategy P&L Over Time',
                 fontsize=16, fontweight='bold', pad=20)

    # Legend
    ax.legend(fontsize=13, loc='best', framealpha=0.9, shadow=True)

    # Grid
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.tick_params(labelsize=12)
    plt.xticks(rotation=45)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_pnl_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_hedge_activity(data, strategy_df, strategy_name, option_type, output_dir):
    """
    PLOT 3: Show when hedging trades happen

    Top panel: Underlying price
    Bottom panel: Positions (stock + hedge options) with trade markers
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.5]})

    opt_label = option_type.upper()

    # TOP PANEL: Underlying price
    ax1.plot(data['Date'], data['Underlying Value'],
             color='black', linewidth=2.5, label='NIFTY Underlying')
    ax1.set_ylabel('NIFTY Level', fontsize=13, fontweight='bold')
    ax1.set_title(f'{opt_label} - {strategy_name}: Underlying Price & Hedge Activity',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # BOTTOM PANEL: Positions
    ax2_twin = ax2.twinx()

    # Stock position (left y-axis)
    color_stock = '#3498db'  # Blue
    ax2.plot(strategy_df['Date'], strategy_df['Stock_Pos'],
             color=color_stock, linewidth=2.5, label='Stock Position', alpha=0.8)
    ax2.set_ylabel('Stock Position (shares)', fontsize=13, fontweight='bold', color=color_stock)
    ax2.tick_params(axis='y', labelcolor=color_stock, labelsize=11)

    # Mark stock trades
    stock_trades = strategy_df[strategy_df['Stock_Trade'] > 0]
    if len(stock_trades) > 0:
        ax2.scatter(stock_trades['Date'], stock_trades['Stock_Pos'],
                    color=color_stock, s=100, marker='o', edgecolors='black',
                    linewidths=1.5, alpha=0.9, zorder=5, label='Stock Rebalance')

    # Hedge option position (right y-axis)
    color_hedge = '#e67e22'  # Orange
    ax2_twin.plot(strategy_df['Date'], strategy_df['Hedge_Opt_Pos'],
                  color=color_hedge, linewidth=2.5, label='Hedge Option Position',
                  linestyle='--', alpha=0.8)
    ax2_twin.set_ylabel('Hedge Option Contracts', fontsize=13, fontweight='bold', color=color_hedge)
    ax2_twin.tick_params(axis='y', labelcolor=color_hedge, labelsize=11)

    # Mark hedge option trades
    hedge_trades = strategy_df[strategy_df['Hedge_Trade'] > 0]
    if len(hedge_trades) > 0:
        ax2_twin.scatter(hedge_trades['Date'], hedge_trades['Hedge_Opt_Pos'],
                         color=color_hedge, s=100, marker='s', edgecolors='black',
                         linewidths=1.5, alpha=0.9, zorder=5, label='Option Hedge Trade')

    # Labels
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45, labelsize=11)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_{strategy_name.lower().replace(" ", "_").replace("-", "_")}_activity.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_option_price_vs_underlying(data, option_type, output_dir):
    """
    PLOT 4: Option price vs underlying (not time) - shows option's price sensitivity

    Scatter plot with smooth fitted curve
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    opt_label = option_type.upper()
    strike = data['Strike Price'].iloc[0]

    # Scatter plot
    color_main = '#e74c3c' if option_type == 'call' else '#3498db'
    ax.scatter(data['Underlying Value'], data['Settle Price'],
               color=color_main, s=80, alpha=0.6, edgecolors='black', linewidths=0.5,
               label=f'{opt_label} Settle Price')

    # Fit smooth curve
    try:
        sorted_data = data.sort_values('Underlying Value')
        spline = UnivariateSpline(sorted_data['Underlying Value'],
                                  sorted_data['Settle Price'], s=10, k=3)
        x_smooth = np.linspace(sorted_data['Underlying Value'].min(),
                               sorted_data['Underlying Value'].max(), 200)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, color='black', linewidth=3,
                linestyle='--', label='Fitted Curve', alpha=0.8)
    except:
        print("  (Could not fit smooth curve)")

    # Strike line
    ax.axvline(x=strike, color='red', linestyle=':', linewidth=2,
               alpha=0.7, label=f'Strike = {strike}')

    # Labels
    ax.set_xlabel('NIFTY Underlying Level', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{opt_label} Option Price (₹)', fontsize=14, fontweight='bold')
    ax.set_title(f'{opt_label} Option: How Price Responds to Underlying Movement',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle=':')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    filename = f'{output_dir}/{option_type}_price_vs_underlying.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def create_summary_statistics(strategies_dict, option_type):
    """
    Create a clean pandas DataFrame with summary statistics

    Returns DataFrame with columns: Strategy, Final_PnL, StdDev, Max_Drawdown
    """
    summary_data = []

    for name, df in strategies_dict.items():
        final_pnl = df['PnL'].iloc[-1]
        std_dev = df['PnL'].std()
        max_pnl = df['PnL'].max()
        min_pnl = df['PnL'].min()
        max_drawdown = max_pnl - min_pnl

        summary_data.append({
            'Option_Type': option_type.upper(),
            'Strategy': name,
            'Final_PnL': final_pnl,
            'StdDev': std_dev,
            'Max_Drawdown': max_drawdown,
            'Risk_Adjusted_Return': final_pnl / std_dev if std_dev > 0 else 0
        })

    return pd.DataFrame(summary_data)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution - runs simulations and creates teaching-oriented visualizations
    """

    # Configuration
    CALL_MAIN_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    CALL_HEDGE_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026 (2).csv'
    PUT_MAIN_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'
    PUT_HEDGE_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026 (2).csv'

    OUTPUT_DIR = 'teaching_outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_CONTRACTS = 100

    print("\n" + "=" * 70)
    print("OPTIONS HEDGING - TEACHING VERSION")
    print("Simple, Clear Visualizations for Education")
    print("=" * 70)

    all_summaries = []

    # ==========================================================================
    # CALL OPTIONS
    # ==========================================================================
    if os.path.exists(CALL_MAIN_FILE) and os.path.exists(CALL_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("ANALYZING CALL OPTIONS")
        print("=" * 70)

        df_call = load_and_prepare_data(CALL_MAIN_FILE, CALL_HEDGE_FILE, 'call')

        # Run strategies
        call_strategies = {
            'Unhedged': simulate_unhedged(df_call, N_CONTRACTS),
            'Delta': simulate_delta_hedge(df_call, N_CONTRACTS, rebalance_threshold=0.20),
            'Delta-Gamma': simulate_delta_gamma_hedge(df_call, N_CONTRACTS, gamma_hedge_ratio=0.70),
            'Delta-Vega': simulate_vega_hedge(df_call, N_CONTRACTS)
        }

        print("\n" + "=" * 70)
        print("CREATING CALL VISUALIZATIONS")
        print("=" * 70)

        # PLOT 1: Underlying vs Option Prices
        plot_underlying_and_option_prices(df_call, 'call', OUTPUT_DIR)

        # PLOT 2: P&L Comparison
        plot_pnl_comparison(call_strategies, 'call', OUTPUT_DIR)

        # PLOT 3: Hedge Activity (for each hedged strategy)
        plot_hedge_activity(df_call, call_strategies['Delta'], 'Delta Hedge', 'call', OUTPUT_DIR)
        plot_hedge_activity(df_call, call_strategies['Delta-Gamma'], 'Delta-Gamma Hedge', 'call', OUTPUT_DIR)
        plot_hedge_activity(df_call, call_strategies['Delta-Vega'], 'Delta-Vega Hedge', 'call', OUTPUT_DIR)

        # PLOT 4: Option Price vs Underlying
        plot_option_price_vs_underlying(df_call, 'call', OUTPUT_DIR)

        # Summary statistics
        call_summary = create_summary_statistics(call_strategies, 'call')
        all_summaries.append(call_summary)

    # ==========================================================================
    # PUT OPTIONS
    # ==========================================================================
    if os.path.exists(PUT_MAIN_FILE) and os.path.exists(PUT_HEDGE_FILE):
        print("\n" + "=" * 70)
        print("ANALYZING PUT OPTIONS")
        print("=" * 70)

        df_put = load_and_prepare_data(PUT_MAIN_FILE, PUT_HEDGE_FILE, 'put')

        # DIAGNOSTIC: Check PUT delta values
        print("\n" + "=" * 70)
        print("DIAGNOSTIC: PUT DELTA VALUES")
        print("=" * 70)
        print(f"First 5 PUT Deltas: {df_put['Delta'].head().tolist()}")
        print(f"PUT Delta range: [{df_put['Delta'].min():.4f}, {df_put['Delta'].max():.4f}]")
        print(f"First PUT price: ₹{df_put['Settle Price'].iloc[0]:.2f}")
        print(f"First underlying: ₹{df_put['Underlying Value'].iloc[0]:.2f}")
        print(f"PUT Strike: {df_put['Strike Price'].iloc[0]}")
        print("=" * 70)

        # Run strategies
        put_strategies = {
            'Unhedged': simulate_unhedged(df_put, N_CONTRACTS),
            'Delta': simulate_delta_hedge(df_put, N_CONTRACTS, rebalance_threshold=0.20),
            'Delta-Gamma': simulate_delta_gamma_hedge(df_put, N_CONTRACTS, gamma_hedge_ratio=1.0),
            'Delta-Vega': simulate_vega_hedge(df_put, N_CONTRACTS)
        }


        print("\n" + "=" * 70)
        print("CREATING PUT VISUALIZATIONS")
        print("=" * 70)

        # PLOT 1: Underlying vs Option Prices
        plot_underlying_and_option_prices(df_put, 'put', OUTPUT_DIR)

        # PLOT 2: P&L Comparison
        plot_pnl_comparison(put_strategies, 'put', OUTPUT_DIR)

        # PLOT 3: Hedge Activity (for each hedged strategy)
        plot_hedge_activity(df_put, put_strategies['Delta'], 'Delta Hedge', 'put', OUTPUT_DIR)
        plot_hedge_activity(df_put, put_strategies['Delta-Gamma'], 'Delta-Gamma Hedge', 'put', OUTPUT_DIR)
        plot_hedge_activity(df_put, put_strategies['Delta-Vega'], 'Delta-Vega Hedge', 'put', OUTPUT_DIR)

        # PLOT 4: Option Price vs Underlying
        plot_option_price_vs_underlying(df_put, 'put', OUTPUT_DIR)

        # Summary statistics
        put_summary = create_summary_statistics(put_strategies, 'put')
        all_summaries.append(put_summary)

        # ==========================================================================
        # FINAL SUMMARY
        # ==========================================================================
        print("\n" + "=" * 70)
        print("✅ ALL VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print(f"\nOutputs saved to: {OUTPUT_DIR}/\n")

        # Combine and display summary statistics
        if all_summaries:
            final_summary = pd.concat(all_summaries, ignore_index=True)
            print("\n" + "=" * 70)
            print("SUMMARY STATISTICS")
            print("=" * 70)
            print(final_summary.to_string(index=False))
            print("\n")

            # Save to CSV
            summary_file = f'{OUTPUT_DIR}/summary_statistics.csv'
            final_summary.to_csv(summary_file, index=False)
            print(f"✓ Summary saved to: {summary_file}\n")

            return final_summary

            # return None

if __name__ == "__main__":
    main()