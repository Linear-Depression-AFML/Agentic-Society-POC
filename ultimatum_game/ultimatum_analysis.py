import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Set style for academic papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('ultimatum_game_log.csv')

# ============================================================================
# FIGURE 1: Trust Evolution Across Rounds
# ============================================================================

def plot_trust_evolution():
    """
    Plot trust evolution showing P_BTrust and R_BTrust across all games.
    Shows asymmetric decay patterns with sharp drops following rejections.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate average trust scores per round
    trust_by_round = df.groupby('Round').agg({
        'P_BTrust': ['mean', 'std', 'sem'],
        'R_BTrust': ['mean', 'std', 'sem'],
        'P_CTrust': ['mean', 'std', 'sem'],
        'R_CTrust': ['mean', 'std', 'sem']
    })
    
    rounds = trust_by_round.index
    
    # Panel 1: Benevolence Trust
    ax1.plot(rounds, trust_by_round['P_BTrust']['mean'], 'o-', 
             label="P_BTrust (Proposer's trust in Responder)", 
             linewidth=2, markersize=6, color='#E63946')
    ax1.fill_between(rounds,
                     trust_by_round['P_BTrust']['mean'] - trust_by_round['P_BTrust']['sem'],
                     trust_by_round['P_BTrust']['mean'] + trust_by_round['P_BTrust']['sem'],
                     alpha=0.2, color='#E63946')
    
    ax1.plot(rounds, trust_by_round['R_BTrust']['mean'], 's-', 
             label="R_BTrust (Responder's trust in Proposer)", 
             linewidth=2, markersize=6, color='#457B9D')
    ax1.fill_between(rounds,
                     trust_by_round['R_BTrust']['mean'] - trust_by_round['R_BTrust']['sem'],
                     trust_by_round['R_BTrust']['mean'] + trust_by_round['R_BTrust']['sem'],
                     alpha=0.2, color='#457B9D')
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Benevolence Trust Score', fontsize=12)
    ax1.set_title('Benevolence Trust Evolution Across Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, max(rounds) + 0.5)
    ax1.set_ylim(0, 1)
    
    # Highlight trust collapse regions
    ax1.axhspan(0, 0.3, alpha=0.1, color='red', label='Trust Collapse Zone')
    ax1.axhspan(0.7, 1.0, alpha=0.1, color='green', label='High Trust Zone')
    
    # Panel 2: Competence Trust
    ax2.plot(rounds, trust_by_round['P_CTrust']['mean'], 'o-', 
             label="P_CTrust (Proposer's trust in Responder)", 
             linewidth=2, markersize=6, color='#F1A208')
    ax2.fill_between(rounds,
                     trust_by_round['P_CTrust']['mean'] - trust_by_round['P_CTrust']['sem'],
                     trust_by_round['P_CTrust']['mean'] + trust_by_round['P_CTrust']['sem'],
                     alpha=0.2, color='#F1A208')
    
    ax2.plot(rounds, trust_by_round['R_CTrust']['mean'], 's-', 
             label="R_CTrust (Responder's trust in Proposer)", 
             linewidth=2, markersize=6, color='#2A9D8F')
    ax2.fill_between(rounds,
                     trust_by_round['R_CTrust']['mean'] - trust_by_round['R_CTrust']['sem'],
                     trust_by_round['R_CTrust']['mean'] + trust_by_round['R_CTrust']['sem'],
                     alpha=0.2, color='#2A9D8F')
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Competence Trust Score', fontsize=12)
    ax2.set_title('Competence Trust Evolution Across Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, max(rounds) + 0.5)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('Plots/ultimatum_trust_evolution.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved: Plots/ultimatum_trust_evolution.png")
    plt.close()

# ============================================================================
# FIGURE 2: Offer Distribution and Acceptance Patterns
# ============================================================================

def plot_offer_distribution():
    """
    Histogram of offers showing concentration around $10 (50% split) with variance.
    Also shows acceptance rate declining for offers below $8.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Panel 1: Offer Distribution
    ax1.hist(df['Amount Offered'], bins=range(0, 22, 1), 
             edgecolor='black', alpha=0.7, color='#A8DADC')
    
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, 
                label='Fair Split ($10)')
    ax1.axvline(x=df['Amount Offered'].mean(), color='orange', 
                linestyle='--', linewidth=2, 
                label=f'Mean Offer (${df["Amount Offered"].mean():.1f})')
    
    ax1.set_xlabel('Amount Offered ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Offers Across All Games', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Acceptance Rate by Offer Amount
    offer_acceptance = df.groupby('Amount Offered').agg({
        'Decision': ['mean', 'count']
    }).reset_index()
    offer_acceptance.columns = ['Amount Offered', 'Acceptance Rate', 'Count']
    
    # Only show offers with at least 3 occurrences for reliable statistics
    offer_acceptance_filtered = offer_acceptance[offer_acceptance['Count'] >= 3]
    
    ax2.plot(offer_acceptance_filtered['Amount Offered'], 
             offer_acceptance_filtered['Acceptance Rate'] * 100, 
             'o-', linewidth=2, markersize=8, color='#E63946')
    
    ax2.axvline(x=8, color='orange', linestyle='--', alpha=0.5, 
                label='Fairness Threshold (~$8)')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, 
                label='50% Acceptance')
    
    ax2.set_xlabel('Amount Offered ($)', fontsize=12)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_title('Acceptance Rate by Offer Amount', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig('Plots/ultimatum_offer_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved: Plots/ultimatum_offer_distribution.png")
    plt.close()

# ============================================================================
# FIGURE 3: Strategic Behavior Analysis
# ============================================================================

def plot_strategic_analysis():
    """
    Two-panel figure showing:
    - Top: Score difference vs. offer size (Gap Anxiety phenomenon)
    - Bottom: Endgame rejection rates by trust level
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Calculate score difference (Proposer - Responder)
    df['Score_Diff'] = df['Proposer Score (Start)'] - df['Responder Score (Start)']
    
    # Panel 1: Score Difference vs. Offer Size
    scatter = ax1.scatter(df['Score_Diff'], df['Amount Offered'], 
                         alpha=0.5, s=50, c=df['Decision'], 
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(df['Score_Diff'], df['Amount Offered'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Score_Diff'].min(), df['Score_Diff'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear Fit')
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(df['Score_Diff'], df['Amount Offered'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}, p = {p_value:.2f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Decision (0=Reject, 1=Accept)', fontsize=10)
    
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Equal Scores')
    ax1.axhline(y=10, color='blue', linestyle='--', alpha=0.5, label='Fair Split ($10)')
    
    ax1.set_xlabel('Score Difference (Proposer - Responder)', fontsize=12)
    ax1.set_ylabel('Amount Offered ($)', fontsize=12)
    ax1.set_title('Gap Anxiety: Score Difference vs. Offer Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Endgame Rejection Rates by Trust Level
    endgame_df = df[df['Round'] >= 8].copy()  # Rounds 8-10
    
    # Categorize trust levels
    endgame_df['Trust_Category'] = pd.cut(endgame_df['R_BTrust'], 
                                          bins=[0, 0.3, 0.6, 1.0],
                                          labels=['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-1.0)'])
    
    rejection_by_trust = endgame_df.groupby('Trust_Category').agg({
        'Decision': lambda x: (1 - x.mean()) * 100  # Convert to rejection rate
    }).reset_index()
    rejection_by_trust.columns = ['Trust_Category', 'Rejection_Rate']
    
    bars = ax2.bar(rejection_by_trust['Trust_Category'], 
                   rejection_by_trust['Rejection_Rate'],
                   color=['#E63946', '#F1A208', '#2A9D8F'], alpha=0.7, edgecolor='black')
    
    ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, 
                label='40% Baseline (Observed Overall)')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Trust Level (R_BTrust)', fontsize=12)
    ax2.set_ylabel('Rejection Rate (%)', fontsize=12)
    ax2.set_title('Endgame Rejection Rates by Trust Level (Rounds 8-10)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('Plots/ultimatum_strategic_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: Plots/ultimatum_strategic_analysis.png")
    plt.close()

# ============================================================================
# ADDITIONAL ANALYSIS: Statistical Summary
# ============================================================================

def print_statistical_summary():
    """
    Print key statistical findings to support claims in the paper.
    """
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY - ULTIMATUM GAME")
    print("="*70)
    
    # 1. Gap Anxiety Analysis
    print("\n1. Gap Anxiety Phenomenon:")
    df['Score_Diff'] = df['Proposer Score (Start)'] - df['Responder Score (Start)']
    
    # After rejection analysis
    rejected_rounds = df[df['Decision'] == 0]
    if len(rejected_rounds) > 0:
        next_round_data = []
        for _, row in rejected_rounds.iterrows():
            game = row['Game']
            round_num = row['Round']
            if round_num < 10:  # Not the last round
                next_round = df[(df['Game'] == game) & (df['Round'] == round_num + 1)]
                if not next_round.empty:
                    next_round_data.append({
                        'Prev_Offer': row['Amount Offered'],
                        'Next_Offer': next_round.iloc[0]['Amount Offered']
                    })
        
        if next_round_data:
            next_df = pd.DataFrame(next_round_data)
            decreased_count = (next_df['Next_Offer'] < next_df['Prev_Offer']).sum()
            total_count = len(next_df)
            decrease_pct = (decreased_count / total_count) * 100
            
            print(f"   After rejection, offers decreased: {decreased_count}/{total_count} ({decrease_pct:.1f}%)")
            print(f"   Mean offer when trailing: ${df[df['Score_Diff'] < 0]['Amount Offered'].mean():.2f}")
            print(f"   Mean offer when leading: ${df[df['Score_Diff'] > 0]['Amount Offered'].mean():.2f}")
    
    # Score diff correlation
    corr, p_val = stats.pearsonr(df['Score_Diff'], df['Amount Offered'])
    print(f"   Correlation (Score Diff vs Offer): r = {corr:.3f}, p = {p_val:.3f}")
    
    # 2. Lead Protection Strategy
    print("\n2. Lead Protection Strategy:")
    endgame_df = df[df['Round'] >= 8]
    leading_responders = endgame_df[endgame_df['Responder Score (Start)'] > endgame_df['Proposer Score (Start)']]
    fair_offers = leading_responders[(leading_responders['Amount Offered'] >= 8) & 
                                     (leading_responders['Amount Offered'] <= 12)]
    
    if len(fair_offers) > 0:
        rejection_rate = (1 - fair_offers['Decision'].mean()) * 100
        print(f"   Endgame fair offer rejection rate (when leading): {rejection_rate:.1f}%")
        print(f"   Overall acceptance rate: {df['Decision'].mean() * 100:.1f}%")
    
    # 3. Fairness Distribution
    print("\n3. Offer Distribution:")
    print(f"   Mean offer: ${df['Amount Offered'].mean():.2f}")
    print(f"   Median offer: ${df['Amount Offered'].median():.2f}")
    print(f"   Std deviation: ${df['Amount Offered'].std():.2f}")
    print(f"   Offers at fair split ($10): {(df['Amount Offered'] == 10).sum()} ({(df['Amount Offered'] == 10).sum() / len(df) * 100:.1f}%)")
    print(f"   Offers below $8: {(df['Amount Offered'] < 8).sum()} ({(df['Amount Offered'] < 8).sum() / len(df) * 100:.1f}%)")
    
    # 4. Trust Breakdown Analysis
    print("\n4. Trust Breakdown Analysis:")
    
    # Calculate trust drops after rejections
    rejection_trust_drops = []
    for _, row in rejected_rounds.iterrows():
        game = row['Game']
        round_num = row['Round']
        if round_num < 10:
            next_round = df[(df['Game'] == game) & (df['Round'] == round_num + 1)]
            if not next_round.empty:
                btrust_drop = row['P_BTrust'] - next_round.iloc[0]['P_BTrust']
                rejection_trust_drops.append(btrust_drop)
    
    if rejection_trust_drops:
        mean_drop = np.mean(rejection_trust_drops)
        print(f"   Mean P_BTrust drop after rejection: {mean_drop:.3f}")
        print(f"   Max P_BTrust drop after rejection: {max(rejection_trust_drops):.3f}")
    
    # Overall trust decay
    initial_btrust = df[df['Round'] == 1]['P_BTrust'].mean()
    final_btrust = df[df['Round'] == 10]['P_BTrust'].mean()
    print(f"   Initial P_BTrust (Round 1): {initial_btrust:.3f}")
    print(f"   Final P_BTrust (Round 10): {final_btrust:.3f}")
    print(f"   Total decay: {(initial_btrust - final_btrust) / initial_btrust * 100:.1f}%")
    
    # 5. Game Outcomes
    print("\n5. Game Outcomes:")
    games_summary = df.groupby('Game').agg({
        'Proposer Gain': 'sum',
        'Responder Gain': 'sum',
        'Decision': 'sum'  # Total accepted offers
    }).reset_index()
    
    proposer_wins = (games_summary['Proposer Gain'] > games_summary['Responder Gain']).sum()
    responder_wins = (games_summary['Responder Gain'] > games_summary['Proposer Gain']).sum()
    ties = (games_summary['Proposer Gain'] == games_summary['Responder Gain']).sum()
    
    print(f"   Proposer wins: {proposer_wins}")
    print(f"   Responder wins: {responder_wins}")
    print(f"   Ties: {ties}")
    print(f"   Mean accepted offers per game: {games_summary['Decision'].mean():.1f}/10")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\nGenerating Ultimatum Game Analysis Plots...")
    print("="*70)
    
    plot_trust_evolution()
    plot_offer_distribution()
    plot_strategic_analysis()
    print_statistical_summary()
    
    print("\n✓ All figures generated successfully!")
    print("="*70)
