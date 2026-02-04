"""
SIMPLIFIED Financial Engineering Hedging Analysis
Focus: Delta, Delta-Gamma, and Vega Hedging Strategies
Uses: Simplified approach with existing libraries
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: SIMPLE BLACK-SCHOLES CALCULATOR (using scipy)
# ============================================================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Calculate option price using Black-Scholes"""
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate delta"""
    if T <= 0:
        return 1.0 if (S > K and option_type == 'call') else 0.0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)

def bs_gamma(S, K, T, r, sigma):
    """Calculate gamma"""
    if T <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    """Calculate vega (per 1% change)"""
    if T <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def implied_vol_simple(option_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using bisection (simple and robust)"""
    if T <= 0:
        return 0.01
    
    # Bisection method
    vol_low, vol_high = 0.01, 5.0
    
    for _ in range(50):  # 50 iterations is enough
        vol_mid = (vol_low + vol_high) / 2
        price_mid = bs_price(S, K, T, r, vol_mid, option_type)
        
        if abs(price_mid - option_price) < 0.01:
            return vol_mid
        
        if price_mid > option_price:
            vol_high = vol_mid
        else:
            vol_low = vol_mid
    
    return vol_mid

# ============================================================================
# PART 2: SIMPLE HEDGING SIMULATOR
# ============================================================================

class SimpleHedging:
    """Simplified hedging analysis focused on strategies"""
    
    def __init__(self, data_file, strike, expiry, risk_free_rate=0.065):
        """Load data and prepare for analysis"""
        # Load data
        self.df = pd.read_csv(data_file)
        self.df.columns = self.df.columns.str.strip()
        
        # Parse dates
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%b-%Y')
        self.expiry = pd.to_datetime(expiry, format='%d-%b-%Y')
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Calculate time to expiry
        self.df['T'] = (self.expiry - self.df['Date']).dt.days / 365
        self.df['T'] = self.df['T'].clip(lower=0)
        
        self.strike = strike
        self.r = risk_free_rate
        
        print(f"Loaded {len(self.df)} days of data")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Strike: {strike}, Expiry: {expiry}")
    
    def calculate_greeks(self):
        """Calculate IV and Greeks for all dates"""
        print("\nCalculating Greeks...")
        
        self.df['IV'] = 0.0
        self.df['Delta'] = 0.0
        self.df['Gamma'] = 0.0
        self.df['Vega'] = 0.0
        
        for idx, row in self.df.iterrows():
            price = row['Settle Price']
            S = row['Underlying Value']
            T = row['T']
            
            # Calculate IV
            iv = implied_vol_simple(price, S, self.strike, T, self.r, 'call')
            self.df.at[idx, 'IV'] = iv
            
            # Calculate Greeks
            self.df.at[idx, 'Delta'] = bs_delta(S, self.strike, T, self.r, iv, 'call')
            self.df.at[idx, 'Gamma'] = bs_gamma(S, self.strike, T, self.r, iv)
            self.df.at[idx, 'Vega'] = bs_vega(S, self.strike, T, self.r, iv)
        
        print("✓ Greeks calculated!")
        return self.df
    
    def strategy_1_unhedged(self, num_options=100):
        """Strategy 1: No hedging - just hold short call position"""
        print("\n[1] Running UNHEDGED strategy...")
        
        results = []
        initial_premium = self.df.iloc[0]['Settle Price'] * num_options
        
        for idx, row in self.df.iterrows():
            option_value = row['Settle Price'] * num_options
            pnl = initial_premium - option_value
            
            results.append({
                'Date': row['Date'],
                'PnL': pnl,
                'Option_Value': option_value
            })
        
        return pd.DataFrame(results)
    
    def strategy_2_delta_hedge(self, num_options=100):
        """Strategy 2: Delta hedging with underlying"""
        print("[2] Running DELTA HEDGING strategy...")
        
        results = []
        initial_premium = self.df.iloc[0]['Settle Price'] * num_options
        
        # Portfolio state
        cash = initial_premium
        stock_position = 0
        last_delta = 0
        
        for idx, row in self.df.iterrows():
            S = row['Underlying Value']
            delta = row['Delta']
            option_value = row['Settle Price'] * num_options
            
            # On first day or when delta changes significantly
            if idx == 0 or abs(delta - last_delta) > 0.05:
                # Adjust stock position to neutralize delta
                target_stock = -delta * num_options
                stock_to_buy = target_stock - stock_position
                cash -= stock_to_buy * S
                stock_position = target_stock
                last_delta = delta
                rebalanced = True
            else:
                rebalanced = False
            
            # Portfolio value
            portfolio_value = cash + stock_position * S - option_value
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'PnL': pnl,
                'Stock_Position': stock_position,
                'Rebalanced': rebalanced
            })
        
        return pd.DataFrame(results)
    
    def strategy_3_delta_gamma_hedge(self, num_options=100, hedge_strike_offset=200):
        """Strategy 3: Delta-Gamma hedging using another option"""
        print("[3] Running DELTA-GAMMA HEDGING strategy...")
        
        results = []
        hedge_strike = self.strike + hedge_strike_offset
        initial_premium = self.df.iloc[0]['Settle Price'] * num_options
        
        # Portfolio state
        cash = initial_premium
        stock_position = 0
        hedge_options = 0
        
        for idx, row in self.df.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            
            # Main option Greeks
            delta_main = row['Delta']
            gamma_main = row['Gamma']
            option_value = row['Settle Price'] * num_options
            
            # Hedge option Greeks (calculate for different strike)
            hedge_price = bs_price(S, hedge_strike, T, self.r, iv, 'call')
            delta_hedge = bs_delta(S, hedge_strike, T, self.r, iv, 'call')
            gamma_hedge = bs_gamma(S, hedge_strike, T, self.r, iv)
            
            # First day: establish hedge
            if idx == 0:
                # Neutralize gamma
                if gamma_hedge > 0:
                    hedge_options = (gamma_main * num_options) / gamma_hedge
                    cash -= hedge_options * hedge_price
                
                # Neutralize delta
                portfolio_delta = -delta_main * num_options + delta_hedge * hedge_options
                stock_position = -portfolio_delta
                cash -= stock_position * S
            
            # Calculate current portfolio value
            hedge_value = hedge_options * hedge_price
            portfolio_value = cash + stock_position * S + hedge_value - option_value
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'PnL': pnl,
                'Hedge_Options': hedge_options
            })
        
        return pd.DataFrame(results)
    
    def strategy_4_vega_hedge(self, num_options=100, hedge_strike_offset=500):
        """Strategy 4: Vega hedging"""
        print("[4] Running VEGA HEDGING strategy...")
        
        results = []
        hedge_strike = self.strike + hedge_strike_offset
        initial_premium = self.df.iloc[0]['Settle Price'] * num_options
        
        # Portfolio state
        cash = initial_premium
        stock_position = 0
        vega_hedge_options = 0
        
        for idx, row in self.df.iterrows():
            S = row['Underlying Value']
            T = row['T']
            iv = row['IV']
            
            # Main option
            vega_main = row['Vega']
            delta_main = row['Delta']
            option_value = row['Settle Price'] * num_options
            
            # Vega hedge option
            vega_hedge_price = bs_price(S, hedge_strike, T, self.r, iv, 'call')
            vega_hedge_vega = bs_vega(S, hedge_strike, T, self.r, iv)
            vega_hedge_delta = bs_delta(S, hedge_strike, T, self.r, iv, 'call')
            
            # First day: establish hedge
            if idx == 0:
                # Neutralize vega
                if vega_hedge_vega > 0:
                    vega_hedge_options = (vega_main * num_options) / vega_hedge_vega
                    cash -= vega_hedge_options * vega_hedge_price
                
                # Neutralize delta
                portfolio_delta = -delta_main * num_options + vega_hedge_delta * vega_hedge_options
                stock_position = -portfolio_delta
                cash -= stock_position * S
            
            # Portfolio value
            vega_hedge_value = vega_hedge_options * vega_hedge_price
            portfolio_value = cash + stock_position * S + vega_hedge_value - option_value
            pnl = portfolio_value - initial_premium
            
            results.append({
                'Date': row['Date'],
                'PnL': pnl,
                'Vega_Hedge_Options': vega_hedge_options
            })
        
        return pd.DataFrame(results)

# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

def create_simple_plots(df, unhedged, delta_hedged, dg_hedged, vega_hedged):
    """Create simple, clear visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Financial Engineering: Hedging Strategies Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. P&L Comparison
    ax = axes[0, 0]
    ax.plot(unhedged['Date'], unhedged['PnL'], label='Unhedged', linewidth=2)
    ax.plot(delta_hedged['Date'], delta_hedged['PnL'], label='Delta Hedged', linewidth=2)
    ax.plot(dg_hedged['Date'], dg_hedged['PnL'], label='Delta-Gamma', linewidth=2)
    ax.plot(vega_hedged['Date'], vega_hedged['PnL'], label='Vega Hedged', linewidth=2)
    ax.set_title('P&L Comparison', fontweight='bold')
    ax.set_ylabel('P&L (₹)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Greeks Evolution
    ax = axes[0, 1]
    ax.plot(df['Date'], df['Delta'], label='Delta', linewidth=2)
    ax.set_title('Delta Evolution', fontweight='bold')
    ax.set_ylabel('Delta')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(df['Date'], df['Gamma'], label='Gamma', color='orange', linewidth=2)
    ax2.set_ylabel('Gamma', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # 3. Final P&L Bar Chart
    ax = axes[0, 2]
    strategies = ['Unhedged', 'Delta\nHedged', 'Delta-Gamma\nHedged', 'Vega\nHedged']
    final_pnl = [
        unhedged['PnL'].iloc[-1],
        delta_hedged['PnL'].iloc[-1],
        dg_hedged['PnL'].iloc[-1],
        vega_hedged['PnL'].iloc[-1]
    ]
    colors = ['green' if x > 0 else 'red' for x in final_pnl]
    ax.bar(strategies, final_pnl, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Final P&L Comparison', fontweight='bold')
    ax.set_ylabel('Final P&L (₹)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Risk Comparison (Std Dev)
    ax = axes[1, 0]
    pnl_std = [
        unhedged['PnL'].std(),
        delta_hedged['PnL'].std(),
        dg_hedged['PnL'].std(),
        vega_hedged['PnL'].std()
    ]
    ax.bar(strategies, pnl_std, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], 
           alpha=0.7, edgecolor='black')
    ax.set_title('Risk Comparison (P&L Volatility)', fontweight='bold')
    ax.set_ylabel('P&L Std Dev (₹)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Underlying vs Option Price
    ax = axes[1, 1]
    ax.plot(df['Date'], df['Underlying Value'], label='NIFTY', linewidth=2, color='blue')
    ax.set_ylabel('NIFTY Price (₹)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title('NIFTY vs Option Price', fontweight='bold')
    
    ax2 = ax.twinx()
    ax2.plot(df['Date'], df['Settle Price'], label='Option', linewidth=2, color='red')
    ax2.set_ylabel('Option Price (₹)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)
    
    # 6. Implied Volatility
    ax = axes[1, 2]
    ax.plot(df['Date'], df['IV'] * 100, linewidth=2, color='purple')
    ax.set_title('Implied Volatility', fontweight='bold')
    ax.set_ylabel('IV (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/simple_hedging_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved!")

# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - SIMPLE and CLEAR"""
    
    print("="*70)
    print("SIMPLIFIED FINANCIAL ENGINEERING HEDGING ANALYSIS")
    print("="*70)
    
    # Initialize
    hedger = SimpleHedging(
        data_file='/mnt/user-data/uploads/OPTIDX_NIFTY_CE_04-Nov-2025_TO_04-Feb-2026__2_.csv',
        strike=26000,
        expiry='24-Feb-2026'
    )
    
    # Calculate Greeks
    df = hedger.calculate_greeks()
    
    print(f"\nLatest Greeks (as of {df['Date'].iloc[-1].strftime('%Y-%m-%d')}):")
    print(f"  Delta:  {df['Delta'].iloc[-1]:.4f}")
    print(f"  Gamma:  {df['Gamma'].iloc[-1]:.6f}")
    print(f"  Vega:   {df['Vega'].iloc[-1]:.2f}")
    print(f"  IV:     {df['IV'].iloc[-1]*100:.2f}%")
    
    # Run strategies
    print("\n" + "="*70)
    print("RUNNING HEDGING STRATEGIES")
    print("="*70)
    
    unhedged = hedger.strategy_1_unhedged()
    delta_hedged = hedger.strategy_2_delta_hedge()
    dg_hedged = hedger.strategy_3_delta_gamma_hedge()
    vega_hedged = hedger.strategy_4_vega_hedge()
    
    # Results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    results = {
        'Strategy': ['Unhedged', 'Delta Hedged', 'Delta-Gamma', 'Vega Hedged'],
        'Final P&L': [
            f"₹{unhedged['PnL'].iloc[-1]:,.0f}",
            f"₹{delta_hedged['PnL'].iloc[-1]:,.0f}",
            f"₹{dg_hedged['PnL'].iloc[-1]:,.0f}",
            f"₹{vega_hedged['PnL'].iloc[-1]:,.0f}"
        ],
        'Risk (Std Dev)': [
            f"₹{unhedged['PnL'].std():,.0f}",
            f"₹{delta_hedged['PnL'].std():,.0f}",
            f"₹{dg_hedged['PnL'].std():,.0f}",
            f"₹{vega_hedged['PnL'].std():,.0f}"
        ]
    }
    
    summary_df = pd.DataFrame(results)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv('/mnt/user-data/outputs/simple_summary.csv', index=False)
    df.to_csv('/mnt/user-data/outputs/simple_greeks.csv', index=False)
    unhedged.to_csv('/mnt/user-data/outputs/simple_unhedged.csv', index=False)
    delta_hedged.to_csv('/mnt/user-data/outputs/simple_delta_hedged.csv', index=False)
    dg_hedged.to_csv('/mnt/user-data/outputs/simple_dg_hedged.csv', index=False)
    vega_hedged.to_csv('/mnt/user-data/outputs/simple_vega_hedged.csv', index=False)
    
    print("\n✓ Results saved to CSV files")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_simple_plots(df, unhedged, delta_hedged, dg_hedged, vega_hedged)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Insight:")
    print("Hedging reduces risk but costs money. In this case, the market")
    print("moved favorably, so unhedged was most profitable. But if the")
    print("market had moved against us, hedging would have saved us!")
    print("="*70)

if __name__ == "__main__":
    main()
