"""
FINANCIAL ENGINEERING: STRATEGY COMPARISON
1. Delta Only (Stock)
2. Delta-Gamma (Option + Stock)
3. Delta-Vega (Option + Stock)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mibian
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# --- 1. SETUP & GREEKS ---
class BlackScholes:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S, self.K, self.T = float(S), float(K), max(float(T), 0.0001)
        self.r, self.sigma = float(r), max(float(sigma), 0.001)
        self.type = option_type.lower()
        self.bs = mibian.BS([self.S, self.K, self.r * 100, self.T * 365], volatility=self.sigma * 100)

    def delta(self): return self.bs.callDelta if self.type == 'call' else self.bs.putDelta

    def gamma(self): return self.bs.gamma

    def vega(self): return self.bs.vega


def get_implied_vol(price, S, K, T, r, type, fallback=0.20):
    if T <= 0.001 or price < 0.05: return fallback
    try:
        bs = mibian.BS([S, K, r * 100, T * 365], callPrice=price if type == 'call' else None,
                       putPrice=price if type == 'put' else None)
        return max(0.01, bs.impliedVolatility / 100)
    except:
        return fallback


def prepare_data(main_file, hedge_file, opt_type):
    df = pd.merge(pd.read_csv(main_file), pd.read_csv(hedge_file).rename(columns={'Settle Price': 'Hedge_Price'}),
                  on='Date')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    K_main = df['Strike Price'].iloc[0]
    K_hedge = K_main + 200 if opt_type == 'call' else K_main - 200
    expiry = pd.to_datetime(df['Expiry'].iloc[0])

    df['T'] = (expiry - df['Date']).dt.days / 365.0
    df['T'] = df['T'].clip(lower=0.0001)

    # Calculate Greeks
    print(f"Calculating Greeks for {opt_type}...")
    for i, row in df.iterrows():
        iv = get_implied_vol(row['Settle Price'], row['Underlying Value'], K_main, row['T'], 0.065, opt_type)
        bs = BlackScholes(row['Underlying Value'], K_main, row['T'], 0.065, iv, opt_type)
        df.at[i, 'Delta'] = bs.delta()
        df.at[i, 'Gamma'] = bs.gamma()
        df.at[i, 'Vega'] = bs.vega()

        h_iv = get_implied_vol(row['Hedge_Price'], row['Underlying Value'], K_hedge, row['T'], 0.065, opt_type)
        bs_h = BlackScholes(row['Underlying Value'], K_hedge, row['T'], 0.065, h_iv, opt_type)
        df.at[i, 'H_Delta'] = bs_h.delta()
        df.at[i, 'H_Gamma'] = bs_h.gamma()
        df.at[i, 'H_Vega'] = bs_h.vega()
    return df


# --- 2. SIMULATION LOGIC ---

def run_simulation(df, mode='delta', N=100):
    cash = df.iloc[0]['Settle Price'] * N
    stock_pos = 0
    hedge_pos = 0
    results = []

    for i, row in df.iterrows():
        # A. Determine Hedge Position (Options)
        target_hedge = 0
        if mode == 'gamma':
            if row['H_Gamma'] > 1e-6:
                target_hedge = -1 * (-1 * row['Gamma'] * N) / row['H_Gamma']
        elif mode == 'vega':
            if row['H_Vega'] > 1e-6:
                target_hedge = -1 * (-1 * row['Vega'] * N) / row['H_Vega']

        # Cap hedge size to avoid explosion
        target_hedge = np.clip(target_hedge, 0, N * 3)

        # Execute Option Trade
        h_trade = target_hedge - hedge_pos
        cash -= (h_trade * row['Hedge_Price'])
        hedge_pos = target_hedge

        # B. Determine Stock Position (Delta)
        # Net Delta must be 0
        short_delta = -1 * row['Delta'] * N
        hedge_delta = hedge_pos * row['H_Delta']
        target_stock = -1 * (short_delta + hedge_delta)

        # Execute Stock Trade
        s_trade = target_stock - stock_pos
        cash -= (s_trade * row['Underlying Value'])
        stock_pos = target_stock

        # C. PnL
        pnl = cash + (stock_pos * row['Underlying Value']) + \
              (hedge_pos * row['Hedge_Price']) - (row['Settle Price'] * N)

        results.append({'Date': row['Date'], 'PnL': pnl})

    return pd.DataFrame(results)


# --- 3. MAIN & PLOTTING ---

# UPDATE FILES HERE
CALL_MAIN = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026.csv'
CALL_HEDGE = 'Data/OPTIDX_NIFTY_CE_08-Nov-2025_TO_08-Feb-2026 (2).csv'

df_call = prepare_data(CALL_MAIN, CALL_HEDGE, 'call')

# Run 3 Strategies
res_delta = run_simulation(df_call, mode='delta')
res_gamma = run_simulation(df_call, mode='gamma')
res_vega = run_simulation(df_call, mode='vega')

# Plot
plt.figure(figsize=(12, 7))
plt.plot(res_delta['Date'], res_delta['PnL'], label='Delta Only', linestyle=':')
plt.plot(res_gamma['Date'], res_gamma['PnL'], label='Delta-Gamma', linewidth=2)
plt.plot(res_vega['Date'], res_vega['PnL'], label='Delta-Vega', linestyle='--')
plt.title('Comparison: Which Hedge Worked Best?')
plt.legend()
plt.ylabel('PnL')
plt.savefig('strategy_comparison.png')
print("Comparison Chart Saved!")