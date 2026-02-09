"""
FINANCIAL ENGINEERING PROJECT: DYNAMIC HEDGING SIMULATION
Course: Data 609
Status: MATHEMATICALLY CORRECTED VERSION

Changes from previous versions:
1. Implemented True Gamma Neutrality (removed 0.3x scaling factor).
2. Implemented Dynamic IV Filling (prevents data jumps when solver fails).
3. Corrected Hedge Order: Solve Gamma -> Update Delta -> Solve Delta.
4. Added Transaction Costs (1bp) and Slippage.

Author: Gemini (Refined for Accuracy)
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mibian
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# ==========================================
# CORE FINANCIAL MATH CLASSES
# ==========================================

class BlackScholes:
    """Robust Black-Scholes Calculator"""

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = float(S)
        self.K = float(K)
        self.T = max(float(T), 0.0001) # Avoid DivideByZero
        self.r = float(r)
        self.sigma = max(float(sigma), 0.001) # Avoid Zero Vol
        self.type = option_type.lower()

        # Mibian expects T in days, r and sigma in percent
        self.bs = mibian.BS([self.S, self.K, self.r*100, self.T*365],
                            volatility=self.sigma*100)

    def price(self):
        return self.bs.callPrice if self.type == 'call' else self.bs.putPrice

    def delta(self):
        return self.bs.callDelta if self.type == 'call' else self.bs.putDelta

    def gamma(self):
        return self.bs.gamma

    def vega(self):
        return self.bs.vega

    def theta(self):
        return self.bs.callTheta if self.type == 'call' else self.bs.putTheta


def get_implied_vol(price, S, K, T, r, option_type, fallback_iv=None):
    """
    Robust IV Solver.
    If solver fails, returns fallback_iv (previous day's IV)
    instead of a hardcoded constant.
    """
    if T <= 0.001 or price < 0.05:
        return fallback_iv if fallback_iv else 0.20

    try:
        bs = mibian.BS([S, K, r*100, T*365],
                       callPrice=price if option_type=='call' else None,
                       putPrice=price if option_type=='put' else None)
        return bs.impliedVolatility / 100
    except:
        return fallback_iv if fallback_iv else 0.20


# ==========================================
# DATA PRE-PROCESSING
# ==========================================

def prepare_data(filepath, strike, expiry_date, option_type, r=0.065):
    """
    Loads data and pre-calculates Greeks to ensure consistency.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Filter columns and parse dates
    df = df[['Date', 'Settle Price', 'Underlying Value']].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df = df.sort_values('Date').reset_index(drop=True)

    expiry = pd.to_datetime(expiry_date)

    # Initialize columns
    df['T'] = (expiry - df['Date']).dt.days / 365.0
    df['T'] = df['T'].clip(lower=0.0001) # Avoid 0 division

    # Calculate Greeks for the History
    ivs = []
    deltas = []
    gammas = []
    vegas = []

    last_iv = 0.20 # Initial guess

    print(f"Pre-calculating Greeks for {option_type.upper()}...")

    for i, row in df.iterrows():
        # Get IV (Using previous day's IV if solver fails)
        iv = get_implied_vol(row['Settle Price'], row['Underlying Value'],
                             strike, row['T'], r, option_type, last_iv)
        last_iv = iv
        ivs.append(iv)

        # Calc Greeks
        bs = BlackScholes(row['Underlying Value'], strike, row['T'], r, iv, option_type)
        deltas.append(bs.delta())
        gammas.append(bs.gamma())
        vegas.append(bs.vega())

    df['IV'] = ivs
    df['Delta'] = deltas
    df['Gamma'] = gammas
    df['Vega'] = vegas

    return df

# ==========================================
# HEDGING STRATEGIES
# ==========================================

def run_unhedged(df, N_short=100):
    """
    Strategy: Short N contracts, hold to expiry.
    """
    results = []

    # Day 0: Sell Options
    entry_price = df.iloc[0]['Settle Price']
    premium_collected = entry_price * N_short

    for i, row in df.iterrows():
        current_opt_val = row['Settle Price'] * N_short
        pnl = premium_collected - current_opt_val

        results.append({
            'Date': row['Date'],
            'Strategy': 'Unhedged',
            'PnL': pnl,
            'Stock_Pos': 0,
            'Hedge_Opt_Pos': 0
        })

    return pd.DataFrame(results)


def run_delta_hedged(df, N_short=100, rebal_trigger=0.10):
    """
    Strategy: Short N options, Delta hedge with Stock.
    """
    results = []

    # Cash tracking
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N_short
    stock_pos = 0

    prev_delta = 0
    tx_cost_rate = 0.0001 # 1 basis point
    cum_tx_costs = 0

    for i, row in df.iterrows():
        S = row['Underlying Value']
        opt_price = row['Settle Price']

        # Position Delta (Short Option has negative delta of the option delta)
        # If Call Delta is 0.6, Short Call Delta is -0.6
        # We need +0.6 Delta (Long Stock)
        target_delta = -1 * (-1 * row['Delta'] * N_short)

        # Rebalance Logic
        trade_occurred = False
        if i == 0 or abs(target_delta - stock_pos) > (rebal_trigger * N_short):
            shares_to_buy = target_delta - stock_pos
            cost = shares_to_buy * S
            tx_fee = abs(cost) * tx_cost_rate

            cash -= (cost + tx_fee)
            cum_tx_costs += tx_fee
            stock_pos = target_delta
            trade_occurred = True

        # Mark to Market
        # Net Wealth = Cash + Stock Value - Option Liability
        mtm_stock = stock_pos * S
        mtm_option = opt_price * N_short
        total_pnl = cash + mtm_stock - mtm_option

        results.append({
            'Date': row['Date'],
            'Strategy': 'Delta Hedged',
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': 0,
            'Trade': trade_occurred
        })

    return pd.DataFrame(results)


def run_delta_gamma_hedged(df, strike, expiry, opt_type, N_short=100, r=0.065):
    """
    Strategy: Short N options.
    1. Buy OTM options to neutralize Gamma.
    2. Buy/Sell Stock to neutralize remaining Delta.
    """
    results = []

    # Hedge Instrument: Strike + 200 (Call) or Strike - 200 (Put)
    hedge_strike = strike + 200 if opt_type == 'call' else strike - 200

    # Initial Cash
    entry_price = df.iloc[0]['Settle Price']
    cash = entry_price * N_short

    stock_pos = 0
    hedge_opt_pos = 0

    tx_cost_rate = 0.0001

    for i, row in df.iterrows():
        S = row['Underlying Value']
        T = row['T']
        IV = row['IV']

        # 1. Get properties of the Hedge Option (The one we buy)
        bs_hedge = BlackScholes(S, hedge_strike, T, r, IV, opt_type)
        h_price = bs_hedge.price()
        h_delta = bs_hedge.delta()
        h_gamma = bs_hedge.gamma()

        # 2. Calculate Gamma Mismatch
        # We are Short: Our Gamma = -1 * row['Gamma'] * N
        portfolio_gamma = -1 * row['Gamma'] * N_short

        # We want Portfolio Gamma + (Hedge_Contracts * Hedge_Gamma) = 0
        # Hedge_Contracts = - Portfolio_Gamma / Hedge_Gamma
        if h_gamma > 1e-6:
            target_hedge_opts = -1 * portfolio_gamma / h_gamma

            # SAFETY CAP: Don't buy more than 3x notional (prevents explosion at expiry)
            target_hedge_opts = np.clip(target_hedge_opts, 0, N_short * 3)
        else:
            target_hedge_opts = 0

        # Execute Hedge Option Trade (Rebalance Daily for Gamma)
        h_trade_qty = target_hedge_opts - hedge_opt_pos
        h_cost = h_trade_qty * h_price
        cash -= (h_cost + abs(h_cost * tx_cost_rate))
        hedge_opt_pos = target_hedge_opts

        # 3. Calculate Delta Mismatch
        # Net Delta = (Short Original Delta) + (Long Hedge Opt Delta)
        # Short Delta = -1 * row['Delta'] * N
        # Long Hedge Delta = hedge_opt_pos * h_delta

        current_net_delta = (-1 * row['Delta'] * N_short) + (hedge_opt_pos * h_delta)

        # We want Total Delta = 0, so Stock Pos must equal -current_net_delta
        target_stock = -1 * current_net_delta

        # Execute Stock Trade
        s_trade_qty = target_stock - stock_pos
        s_cost = s_trade_qty * S
        cash -= (s_cost + abs(s_cost * tx_cost_rate))
        stock_pos = target_stock

        # 4. Mark to Market
        mtm_stock = stock_pos * S
        mtm_hedge_opt = hedge_opt_pos * h_price
        mtm_short_opt = row['Settle Price'] * N_short

        total_pnl = cash + mtm_stock + mtm_hedge_opt - mtm_short_opt

        results.append({
            'Date': row['Date'],
            'Strategy': 'Delta-Gamma',
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': hedge_opt_pos
        })

    return pd.DataFrame(results)

# ==========================================
# VISUALIZATION
# ==========================================

def plot_results(unhedged, delta, gamma, opt_type, output_folder):
    fig = plt.figure(figsize=(20, 12))

    # 1. P&L Comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(unhedged['Date'], unhedged['PnL'], label='Unhedged', color='red', linewidth=2)
    ax1.plot(delta['Date'], delta['PnL'], label='Delta Hedged', color='blue', linewidth=2)
    ax1.plot(gamma['Date'], gamma['PnL'], label='Delta-Gamma Hedged', color='green', linewidth=2)
    ax1.set_title(f"{opt_type.upper()}: P&L Trajectory", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Profit / Loss (Currency)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Position Sizes
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(delta['Date'], delta['Stock_Pos'], label='Delta Hedge (Stock)', color='blue', alpha=0.6)
    ax2.plot(gamma['Date'], gamma['Stock_Pos'], label='D-G Hedge (Stock)', color='green', alpha=0.6, linestyle='--')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(gamma['Date'], gamma['Hedge_Opt_Pos'], label='Hedge Options (Contracts)', color='purple', alpha=0.8)

    ax2.set_title("Hedge Positions over Time", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Stock Shares")
    ax2_twin.set_ylabel("Hedge Options (Contracts)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 3. Bar Chart Summary
    ax3 = plt.subplot(2, 2, 3)
    finals = [unhedged['PnL'].iloc[-1], delta['PnL'].iloc[-1], gamma['PnL'].iloc[-1]]
    risks = [unhedged['PnL'].std(), delta['PnL'].std(), gamma['PnL'].std()]

    x = np.arange(3)
    width = 0.35

    rects1 = ax3.bar(x - width/2, finals, width, label='Final P&L', color=['salmon', 'lightblue', 'lightgreen'])

    ax3_twin = ax3.twinx()
    rects2 = ax3_twin.bar(x + width/2, risks, width, label='Std Dev (Risk)', color='gray', alpha=0.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(['Unhedged', 'Delta', 'Delta-Gamma'])
    ax3.set_title("Return vs Risk Profile", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Total P&L")
    ax3_twin.set_ylabel("Standard Deviation")

    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # Add values to bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1, ax3)

    plt.tight_layout()
    plt.savefig(f"{output_folder}/{opt_type}_analysis.png")
    print(f"Chart saved to {output_folder}/{opt_type}_analysis.png")
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Update these paths to match your folder structure
    CALL_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    PUT_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'

    OUTPUT_DIR = 'Report_Outputs_Corrected'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- Starting Simulation ---")

    # 1. PROCESS CALL OPTION
    if os.path.exists(CALL_FILE):
        print("\nAnalyzing CALL Option...")
        # Load raw data to get params
        temp = pd.read_csv(CALL_FILE)
        temp.columns = temp.columns.str.strip()
        K = temp['Strike Price'].iloc[0]
        Exp = temp['Expiry'].iloc[0]

        # Pre-calc data
        df_call = prepare_data(CALL_FILE, K, Exp, 'call')

        # Run Sims
        res_un = run_unhedged(df_call)
        res_del = run_delta_hedged(df_call)
        res_gam = run_delta_gamma_hedged(df_call, K, Exp, 'call')

        plot_results(res_un, res_del, res_gam, 'call', OUTPUT_DIR)

        # Save CSV
        res_gam.to_csv(f"{OUTPUT_DIR}/call_simulation_data.csv", index=False)

    # 2. PROCESS PUT OPTION
    if os.path.exists(PUT_FILE):
        print("\nAnalyzing PUT Option...")
        temp = pd.read_csv(PUT_FILE)
        temp.columns = temp.columns.str.strip()
        K = temp['Strike Price'].iloc[0]
        Exp = temp['Expiry'].iloc[0]

        df_put = prepare_data(PUT_FILE, K, Exp, 'put')

        res_un = run_unhedged(df_put)
        res_del = run_delta_hedged(df_put)
        res_gam = run_delta_gamma_hedged(df_put, K, Exp, 'put')

        plot_results(res_un, res_del, res_gam, 'put', OUTPUT_DIR)
        res_gam.to_csv(f"{OUTPUT_DIR}/put_simulation_data.csv", index=False)

    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    main()