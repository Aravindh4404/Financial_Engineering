import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def analyze_hedging_costs(filename, strategy_name):
    """
    Decomposes the PnL into 'Naked Short' and 'Hedge Cost'.
    """
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # 1. Calculate the Theoretical Profit if you had NO hedge
    # Short Profit = (Entry Price - Current Price) * 100
    entry_price = df['Main_Price'].iloc[0]
    df['Short_PnL'] = (entry_price - df['Main_Price']) * 100

    # 2. Calculate the Cost of the Hedge
    # Hedge Cost = Total PnL - Short PnL
    df['Hedge_PnL'] = df['PnL'] - df['Short_PnL']

    # Metrics
    final_net = df['PnL'].iloc[-1]
    final_short = df['Short_PnL'].iloc[-1]
    final_hedge = df['Hedge_PnL'].iloc[-1]

    print(f"\n--- {strategy_name} ---")
    print(f"Net Strategy PnL:   {final_net:,.2f}")
    print(f"Unhedged Short PnL: {final_short:,.2f} (Income Source)")
    print(f"Total Hedge Cost:   {final_hedge:,.2f} (Insurance Cost)")

    return df

# Run Analysis
call_df = analyze_hedging_costs('call_real_results.csv', 'Call Strategy')
put_df = analyze_hedging_costs('put_real_results.csv', 'Put Strategy')

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Call Chart
ax1.plot(call_df['Date'], call_df['Short_PnL'], color='gray', linestyle='--', alpha=0.6, label='Unhedged Income')
ax1.plot(call_df['Date'], call_df['Hedge_PnL'], color='red', linestyle=':', linewidth=2, label='Hedge Cost')
ax1.plot(call_df['Date'], call_df['PnL'], color='green', linewidth=3, label='Net Profit')
ax1.set_title('Call Strategy Decomposition', fontsize=14, fontweight='bold')
ax1.legend()

# Put Chart
ax2.plot(put_df['Date'], put_df['Short_PnL'], color='gray', linestyle='--', alpha=0.6, label='Unhedged Income')
ax2.plot(put_df['Date'], put_df['Hedge_PnL'], color='orange', linestyle=':', linewidth=2, label='Hedge Cost')
ax2.plot(put_df['Date'], put_df['PnL'], color='red', linewidth=3, label='Net Profit')
ax2.set_title('Put Strategy Decomposition', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig('hedging_cost_breakdown.png')
print("\nChart saved as hedging_cost_breakdown.png")