import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# --- CONFIGURATION ---
CSV_FILE = 'trust_game_log.csv'
OUTPUT_DPI = 300  # High DPI for research paper quality

# Set a professional style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'  # Use serif fonts for academic look
plt.rcParams['axes.titleweight'] = 'bold'

def load_data():
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILE}. Please run the simulation first.")
        return None

def plot_trust_evolution(df):
    """
    Figure 1: Evolution of Trust Scores over Rounds (Average across all games).
    Shows how CTrust and BTrust change for both agents.
    """
    print("Generating Trust Evolution Plot...")
    plt.figure(figsize=(10, 6))
    
    # Group by Round and calculate mean
    avg_data = df.groupby('Round')[['S_CTrust', 'S_BTrust', 'R_CTrust', 'R_BTrust']].mean().reset_index()
    
    # Plot lines
    plt.plot(avg_data['Round'], avg_data['S_CTrust'], label='Sender Competence', marker='o', linestyle='-', color='#1f77b4')
    plt.plot(avg_data['Round'], avg_data['S_BTrust'], label='Sender Benevolence', marker='s', linestyle='--', color='#ff7f0e')
    plt.plot(avg_data['Round'], avg_data['R_CTrust'], label='Receiver Competence', marker='^', linestyle=':', color='#2ca02c')
    plt.plot(avg_data['Round'], avg_data['R_BTrust'], label='Receiver Benevolence', marker='d', linestyle='-.', color='#d62728')

    plt.title('Average Evolution of Trust Scores Over 10 Rounds')
    plt.xlabel('Round Number')
    plt.ylabel('Trust Score (0.0 - 1.0)')
    plt.ylim(0, 1.05)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig('fig1_trust_evolution_avg.png', dpi=OUTPUT_DPI)
    plt.close()

def plot_trust_action_correlation(df):
    """
    Figure 2: Correlation between Sender's Benevolence Trust and Amount Sent.
    Validates the 'Closed-Loop' hypothesis.
    """
    print("Generating Trust-Action Correlation Plot...")
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with regression line
    sns.regplot(
        data=df, 
        x='S_BTrust', 
        y='Amount Sent By Sender', 
        scatter_kws={'alpha':0.5, 's':50, 'color':'#1f77b4'},
        line_kws={'color':'#d62728', 'linewidth':2}
    )
    
    # Calculate correlation coefficient
    corr = df['S_BTrust'].corr(df['Amount Sent By Sender'])
    
    plt.title(f'Sender Action vs. Benevolence Trust (Correlation: {corr:.2f})')
    plt.xlabel('Sender Benevolence Trust (S_BTrust)')
    plt.ylabel('Amount Sent ($)')
    plt.text(0.05, 0.9, f'r = {corr:.2f}', transform=plt.gca().transAxes, fontsize=14, color='red')
    plt.tight_layout()
    plt.savefig('fig2_trust_action_corr.png', dpi=OUTPUT_DPI)
    plt.close()

def plot_game_outcome_comparison(df):
    """
    Figure 3: Total Wealth Accumulation by Role.
    Shows the "Rational Exploiter" finding (Receiver earning more).
    """
    print("Generating Game Outcome Plot...")
    
    # Get the final round for each game
    final_rounds = df.groupby('Game').tail(1)
    
    # Prepare data for bar plot
    total_sender = final_rounds['Final Sender Amount'].sum()
    total_receiver = final_rounds['Final Receiver Amount'].sum()
    
    outcomes = pd.DataFrame({
        'Role': ['Sender', 'Receiver'],
        'Total Wealth': [total_sender, total_receiver]
    })
    
    plt.figure(figsize=(7, 6))
    ax = sns.barplot(x='Role', y='Total Wealth', data=outcomes, palette=['#1f77b4', '#ff7f0e'])
    
    # Add value labels on top of bars
    for i, v in enumerate(outcomes['Total Wealth']):
        ax.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Total Wealth Accumulation by Role (All Games)')
    plt.ylabel('Total Wealth ($)')
    plt.tight_layout()
    plt.savefig('fig3_wealth_outcomes.png', dpi=OUTPUT_DPI)
    plt.close()

def plot_betrayal_case_study(df, game_id=2):
    """
    Figure 4: Case Study of a specific game (e.g., Game 2) showing Betrayal.
    Overlays Sender's Trust with Receiver's Return Ratio.
    """
    print(f"Generating Case Study Plot for Game {game_id}...")
    
    game_data = df[df['Game'] == game_id].copy()
    if game_data.empty:
        print(f"Warning: Game {game_id} not found in data.")
        return

    # Calculate Return Ratio (Receiver Action Fairness)
    # Pot size is Amount Sent * 2. Return Ratio = Amount Sent Back / Pot Size.
    game_data['Pot Size'] = game_data['Amount Sent By Sender'] * 2
    game_data['Return Ratio'] = game_data.apply(
        lambda row: row['Amount Sent By Receiver'] / row['Pot Size'] if row['Pot Size'] > 0 else 0, axis=1
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Sender's Benevolence Trust on Left Y-Axis
    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Sender Benevolence Trust (S_BTrust)', color=color)
    ax1.plot(game_data['Round'], game_data['S_BTrust'], color=color, marker='o', linewidth=2, label='S_BTrust')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    ax1.grid(False) # Turn off grid for secondary axis clarity

    # Create a second y-axis for the Bar Chart (Return Ratio)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Receiver Return Ratio (%)', color=color)  
    # Plot as bars
    ax2.bar(game_data['Round'], game_data['Return Ratio'], color=color, alpha=0.3, label='Return Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)
    
    # Add a reference line for "Fairness" (0.5 or 50%)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Fair Split (0.5)')

    plt.title(f'Case Study (Game {game_id}): Impact of Return Ratio on Trust')
    fig.tight_layout()  
    plt.savefig(f'fig4_case_study_game_{game_id}.png', dpi=OUTPUT_DPI)
    plt.close()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_trust_evolution(df)
        plot_trust_action_correlation(df)
        plot_game_outcome_comparison(df)
        
        # Check which game shows a good betrayal (Game 2 in your logs was good)
        # You can change the game_id here to visualize different games
        plot_betrayal_case_study(df, game_id=2) 
        
        print("\nDone! All plots saved as PNG files.")