import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the visual style for professional reports
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})


def load_and_prep_data(filename):
    """Loads CSV and converts dates for plotting."""
    try:
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Make sure it is in the same folder.")
        return None


def calculate_metrics(df, name):
    """Calculates key financial metrics for the strategy."""
    if df is None: return None

    final_pnl = df['PnL'].iloc[-1]
    max_pnl = df['PnL'].max()

    # Max Drawdown (Deepest dive from a previous peak)
    df['Peak'] = df['PnL'].cummax()
    df['Drawdown'] = df['PnL'] - df['Peak']
    max_drawdown = df['Drawdown'].min()

    # Volatility of daily PnL changes
    daily_pnl_change = df['PnL'].diff().std()

    return {
        "Strategy": name,
        "Final PnL": f"{final_pnl:,.2f}",
        "Max Profit": f"{max_pnl:,.2f}",
        "Max Drawdown": f"{max_drawdown:,.2f}",
        "PnL Volatility": f"{daily_pnl_change:,.2f}"
    }


def plot_combined_pnl(call_df, put_df):
    """Generates the comparison chart between Call and Put strategies."""
    plt.figure(figsize=(12, 6))

    plt.plot(call_df['Date'], call_df['PnL'], label='Short Call Strategy (Delta-Gamma)',
             color='#2ca02c', linewidth=2.5)  # Green
    plt.plot(put_df['Date'], put_df['PnL'], label='Short Put Strategy (Delta-Gamma)',
             color='#d62728', linewidth=2.5)  # Red

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Cumulative PnL Trajectory: Call vs Put Strategies', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit/Loss')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig('1_combined_pnl_trajectory.png', dpi=300)
    print("Saved: 1_combined_pnl_trajectory.png")
    plt.close()


def plot_deep_dive(df, option_type, pnl_color):
    """Generates a detailed dual-axis chart for a specific strategy."""
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot PnL on Left Axis
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative PnL', color=pnl_color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(df['Date'], df['PnL'], color=pnl_color, linewidth=3, label='Cumulative PnL')
    ax1.tick_params(axis='y', labelcolor=pnl_color)
    ax1.axhline(0, color='black', linestyle='-', alpha=0.2)

    # Plot Positions on Right Axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Hedge Positions', color='#1f77b4', fontsize=12, fontweight='bold')

    line2 = ax2.plot(df['Date'], df['Stock_Pos'], color='#1f77b4', linestyle='--',
                     alpha=0.8, label='Stock Hedge (Shares)')
    line3 = ax2.plot(df['Date'], df['Hedge_Opt_Pos'], color='purple', linestyle=':',
                     linewidth=2.5, label='Hedge Options (Contracts)')

    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Combined Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9)

    plt.title(f'Deep Dive: {option_type} Strategy Dynamics\n(PnL vs Hedge Adjustments)',
              fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'2_{option_type.lower()}_deep_dive.png', dpi=300)
    print(f"Saved: 2_{option_type.lower()}_deep_dive.png")
    plt.close()


# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    call_df = load_and_prep_data('call_simulation_data.csv')
    put_df = load_and_prep_data('put_simulation_data.csv')

    if call_df is not None and put_df is not None:
        # 2. Generate Metrics
        print("\n--- PERFORMANCE METRICS ---")
        metrics = [
            calculate_metrics(call_df, "Short Call"),
            calculate_metrics(put_df, "Short Put")
        ]
        metrics_df = pd.DataFrame(metrics)
        print(metrics_df.to_string(index=False))

        # Save metrics to text file
        with open('performance_metrics.txt', 'w') as f:
            f.write(metrics_df.to_string(index=False))

        # 3. Generate Charts
        print("\n--- GENERATING CHARTS ---")
        plot_combined_pnl(call_df, put_df)
        plot_deep_dive(call_df, "Call", '#2ca02c')  # Green for Call
        plot_deep_dive(put_df, "Put", '#d62728')  # Red for Put

        print("\nDone! Check your folder for the PNG images and text report.")