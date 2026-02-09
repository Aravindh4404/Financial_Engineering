"""
FINANCIAL ENGINEERING PROJECT: REAL MARKET DATA HEDGING
Status: FINAL - Uses Real Market Prices for Hedge Options

Description:
This script simulates Delta-Gamma hedging using REAL historical prices
for both the short option and the long hedge option. This removes
theoretical pricing errors and makes the P&L 100% realistic.

Instructions:
1. Ensure you have 4 CSV files in a folder named 'Data'.
2. Update the 'FILES CONFIGURATION' section at the bottom if needed.
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


# ==========================================
# 1. HELPER CLASSES (For Greeks Calculation)
# ==========================================

class BlackScholes:
    """
    Used ONLY to calculate Greeks (Delta/Gamma) to determine hedge ratios.
    Prices are taken directly from the CSV files.
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = float(S)
        self.K = float(K)
        self.T = max(float(T), 0.0001)
        self.r = float(r)
        self.sigma = max(float(sigma), 0.001)
        self.type = option_type.lower()
        self.bs = mibian.BS([self.S, self.K, self.r * 100, self.T * 365],
                            volatility=self.sigma * 100)

    def delta(self):
        return self.bs.callDelta if self.type == 'call' else self.bs.putDelta

    def gamma(self):
        return self.bs.gamma


def get_implied_vol(price, S, K, T, r, option_type, fallback_iv=0.20):
    """
    Reverse-engineers IV from the market price to calculate accurate Greeks.
    """
    if T <= 0.001 or price < 0.05: return fallback_iv
    try:
        bs = mibian.BS([S, K, r * 100, T * 365],
                       callPrice=price if option_type == 'call' else None,
                       putPrice=price if option_type == 'put' else None)
        return max(0.01, bs.impliedVolatility / 100)
    except:
        return fallback_iv


# ==========================================
# 2. DATA LOADING & PRE-PROCESSING
# ==========================================

def prepare_real_data(main_file, hedge_file, option_type, r=0.065):
    """
    Loads and merges the Main Option and Hedge Option files.
    Calculates Greeks for both based on their Real Prices.
    """
    print(f"--- Processing {option_type.upper()} Data ---")

    # A. Load Main File (The one we sold)
    df_main = pd.read_csv(main_file)
    df_main.columns = df_main.columns.str.strip()
    df_main['Date'] = pd.to_datetime(df_main['Date'])

    # B. Load Hedge File (The one we buy)
    df_hedge = pd.read_csv(hedge_file)
    df_hedge.columns = df_hedge.columns.str.strip()
    df_hedge['Date'] = pd.to_datetime(df_hedge['Date'])

    # Rename Hedge Price to avoid confusion
    df_hedge = df_hedge[['Date', 'Settle Price']].rename(columns={'Settle Price': 'Hedge_Price'})

    # C. Merge on Date (Inner Join to ensure we have prices for both)
    df = pd.merge(df_main, df_hedge, on='Date', how='inner')
    df = df.sort_values('Date').reset_index(drop=True)

    # D. Setup Parameters
    strike_main = df['Strike Price'].iloc[0]

    # Auto-detect Hedge Strike (approximate) or calculate strictly
    # (Here we assume the user provided the correct file, but we compute Greeks based on implied logic)
    if option_type == 'call':
        strike_hedge = strike_main + 200
    else:
        strike_hedge = strike_main - 200

    expiry = pd.to_datetime(df['Expiry'].iloc[0])

    # E. Calculate Time to Expiry
    df['T'] = (expiry - df['Date']).dt.days / 365.0
    df['T'] = df['T'].clip(lower=0.0001)

    # F. Calculate Greeks for MAIN Option
    print(f"  Calculating Greeks for Main Strike ({strike_main})...")
    ivs, deltas, gammas = [], [], []
    last_iv = 0.20

    for i, row in df.iterrows():
        iv = get_implied_vol(row['Settle Price'], row['Underlying Value'],
                             strike_main, row['T'], r, option_type, last_iv)
        last_iv = iv

        bs = BlackScholes(row['Underlying Value'], strike_main, row['T'], r, iv, option_type)
        ivs.append(iv)
        deltas.append(bs.delta())
        gammas.append(bs.gamma())

    df['IV'] = ivs
    df['Delta'] = deltas
    df['Gamma'] = gammas

    # G. Calculate Greeks for HEDGE Option (using its Real Price)
    print(f"  Calculating Greeks for Hedge Strike ({strike_hedge})...")
    h_deltas, h_gammas = [], []
    last_h_iv = 0.20

    for i, row in df.iterrows():
        # Get IV from the HEDGE price
        h_iv = get_implied_vol(row['Hedge_Price'], row['Underlying Value'],
                               strike_hedge, row['T'], r, option_type, last_h_iv)
        last_h_iv = h_iv

        bs_h = BlackScholes(row['Underlying Value'], strike_hedge, row['T'], r, h_iv, option_type)
        h_deltas.append(bs_h.delta())
        h_gammas.append(bs_h.gamma())

    df['Hedge_Delta'] = h_deltas
    df['Hedge_Gamma'] = h_gammas

    return df


# ==========================================
# 3. CORE SIMULATION ENGINE
# ==========================================

def run_simulation(df, N_short=100):
    """
    Runs the Delta-Gamma hedging simulation using Real Prices.
    """
    results = []

    # Initial Cash (Premium Received)
    entry_premium = df.iloc[0]['Settle Price'] * N_short
    cash = entry_premium

    stock_pos = 0
    hedge_opt_pos = 0
    tx_cost = 0.0001  # 1 basis point transaction cost

    for i, row in df.iterrows():
        S = row['Underlying Value']

        # --- STEP 1: GAMMA HEDGE ---
        # We are Short Main Option -> We have Negative Gamma
        # We Buy Hedge Option -> We get Positive Gamma
        # Goal: Net Gamma = 0

        short_gamma_risk = -1 * row['Gamma'] * N_short
        long_gamma_per_contract = row['Hedge_Gamma']

        if long_gamma_per_contract > 1e-7:
            # How many contracts to buy?
            target_hedge_opts = -1 * short_gamma_risk / long_gamma_per_contract
            # Safety cap: Don't buy more than 3x the notional size
            target_hedge_opts = np.clip(target_hedge_opts, 0, N_short * 3)
        else:
            target_hedge_opts = 0

        # Execute Hedge Trade (Using REAL PRICE)
        hedge_trade_qty = target_hedge_opts - hedge_opt_pos
        hedge_trade_price = row['Hedge_Price']  # <--- REAL MARKET PRICE

        cost_hedge = hedge_trade_qty * hedge_trade_price
        cash -= (cost_hedge + abs(cost_hedge * tx_cost))
        hedge_opt_pos = target_hedge_opts

        # --- STEP 2: DELTA HEDGE ---
        # Net Delta = (Short Delta) + (Hedge Opt Delta) + (Stock Delta)
        # Goal: Net Delta = 0

        short_delta_risk = -1 * row['Delta'] * N_short
        hedge_pos_delta = hedge_opt_pos * row['Hedge_Delta']

        current_total_delta = short_delta_risk + hedge_pos_delta

        # We need stock to oppose this
        target_stock = -1 * current_total_delta

        # Execute Stock Trade
        stock_trade_qty = target_stock - stock_pos
        cost_stock = stock_trade_qty * S
        cash -= (cost_stock + abs(cost_stock * tx_cost))
        stock_pos = target_stock

        # --- STEP 3: MARK TO MARKET (PnL) ---
        # Value = Cash + Asset Value - Liability Value

        mtm_stock = stock_pos * S
        mtm_hedge = hedge_opt_pos * row['Hedge_Price']  # <--- REAL MARKET PRICE
        mtm_short = row['Settle Price'] * N_short  # <--- REAL MARKET PRICE

        total_pnl = cash + mtm_stock + mtm_hedge - mtm_short

        results.append({
            'Date': row['Date'],
            'PnL': total_pnl,
            'Stock_Pos': stock_pos,
            'Hedge_Opt_Pos': hedge_opt_pos,
            'Main_Price': row['Settle Price'],
            'Hedge_Price': row['Hedge_Price']
        })

    return pd.DataFrame(results)


# ==========================================
# 4. REPORTING
# ==========================================

def save_and_plot(res_call, res_put, output_dir):
    # 1. Save CSVs
    if not res_call.empty:
        res_call.to_csv(f'{output_dir}/call_real_results.csv', index=False)
    if not res_put.empty:
        res_put.to_csv(f'{output_dir}/put_real_results.csv', index=False)

    # 2. Plot
    plt.figure(figsize=(14, 7))

    if not res_call.empty:
        plt.plot(res_call['Date'], res_call['PnL'], label='Call Strategy (Real Data)',
                 color='green', linewidth=2)

    if not res_put.empty:
        plt.plot(res_put['Date'], res_put['PnL'], label='Put Strategy (Real Data)',
                 color='red', linewidth=2)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Delta-Gamma Hedging: Real Market Data Analysis', fontsize=16, fontweight='bold')
    plt.ylabel('Cumulative PnL (Currency)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/Real_Data_Comparison.png', dpi=300)
    print(f"\n✓ Chart Saved: {output_dir}/Real_Data_Comparison.png")


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    # ---------------------------------------------------------
    # FILES CONFIGURATION (UPDATE THESE NAMES!)
    # ---------------------------------------------------------
    # Call Side
    CALL_MAIN_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
    CALL_HEDGE_FILE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026 (2).csv'  # e.g. Strike 24200

    # Put Side
    PUT_MAIN_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026.csv'
    PUT_HEDGE_FILE = 'Data/OPTIDX_NIFTY_PE_08-Nov-2025_TO_08-Feb-2026 (2).csv'  # e.g. Strike 23800
    # ---------------------------------------------------------

    OUTPUT_DIR = 'Real_Data_Output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STARTING REAL DATA SIMULATION")
    print("=" * 60)

    # Run Call Simulation
    res_call = pd.DataFrame()
    if os.path.exists(CALL_MAIN_FILE) and os.path.exists(CALL_HEDGE_FILE):
        df_call = prepare_real_data(CALL_MAIN_FILE, CALL_HEDGE_FILE, 'call')
        res_call = run_simulation(df_call)
    else:
        print(f"⚠️  Skipping Call: Files not found.")

    # Run Put Simulation
    res_put = pd.DataFrame()
    if os.path.exists(PUT_MAIN_FILE) and os.path.exists(PUT_HEDGE_FILE):
        df_put = prepare_real_data(PUT_MAIN_FILE, PUT_HEDGE_FILE, 'put')
        res_put = run_simulation(df_put)
    else:
        print(f"⚠️  Skipping Put: Files not found.")

    # Finalize
    save_and_plot(res_call, res_put, OUTPUT_DIR)
    print("\n✅ Simulation Complete.")


if __name__ == "__main__":
    main()