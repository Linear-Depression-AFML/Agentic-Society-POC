import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TrustGameVisualizer:
    """Visualizer for Trust Game results with comprehensive analysis."""
    
    def __init__(self, csv_path):
        """Initialize with path to CSV file."""
        self.df = pd.read_csv(csv_path)
        self.output_dir = Path("trust_plots")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set consistent style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        print(f"✓ Loaded {len(self.df)} records from {csv_path}")
        print(f"  Games: {self.df['Game'].nunique()}, Max Round: {self.df['Round'].max()}")
        print(f"  Columns: {list(self.df.columns)}")
    
    def plot_trust_evolution_aggregate(self):
        """Plot average trust evolution across all games with confidence intervals."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calculate round-wise statistics
        round_stats = self.df.groupby('Round').agg({
            'S_CTrust': ['mean', 'std'],
            'S_BTrust': ['mean', 'std'],
            'R_CTrust': ['mean', 'std'],
            'R_BTrust': ['mean', 'std']
        }).reset_index()
        
        rounds = round_stats['Round']
        
        # Plot 1: Competence Trust
        ax1 = axes[0]
        ax1.plot(rounds, round_stats['S_CTrust']['mean'], 
                marker='o', linewidth=2.5, label='Sender→Receiver', color='#2E86AB')
        ax1.fill_between(rounds, 
                         round_stats['S_CTrust']['mean'] - round_stats['S_CTrust']['std'],
                         round_stats['S_CTrust']['mean'] + round_stats['S_CTrust']['std'],
                         alpha=0.2, color='#2E86AB')
        
        ax1.plot(rounds, round_stats['R_CTrust']['mean'], 
                marker='s', linewidth=2.5, label='Receiver→Sender', color='#A23B72', linestyle='--')
        ax1.fill_between(rounds,
                         round_stats['R_CTrust']['mean'] - round_stats['R_CTrust']['std'],
                         round_stats['R_CTrust']['mean'] + round_stats['R_CTrust']['std'],
                         alpha=0.2, color='#A23B72')
        
        ax1.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Competence Trust Score', fontsize=12, fontweight='bold')
        ax1.set_title('Average Competence Trust Evolution', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot 2: Benevolence Trust
        ax2 = axes[1]
        ax2.plot(rounds, round_stats['S_BTrust']['mean'], 
                marker='o', linewidth=2.5, label='Sender→Receiver', color='#2E86AB')
        ax2.fill_between(rounds,
                         round_stats['S_BTrust']['mean'] - round_stats['S_BTrust']['std'],
                         round_stats['S_BTrust']['mean'] + round_stats['S_BTrust']['std'],
                         alpha=0.2, color='#2E86AB')
        
        ax2.plot(rounds, round_stats['R_BTrust']['mean'], 
                marker='s', linewidth=2.5, label='Receiver→Sender', color='#A23B72', linestyle='--')
        ax2.fill_between(rounds,
                         round_stats['R_BTrust']['mean'] - round_stats['R_BTrust']['std'],
                         round_stats['R_BTrust']['mean'] + round_stats['R_BTrust']['std'],
                         alpha=0.2, color='#A23B72')
        
        ax2.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Benevolence Trust Score', fontsize=12, fontweight='bold')
        ax2.set_title('Average Benevolence Trust Evolution', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9, fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_aggregate.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_individual_games(self, max_games=5):
        """Plot trust evolution for individual games (limited to avoid legend clutter)."""
        games = sorted(self.df['Game'].unique())[:max_games]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(games)))
        
        for idx, game_num in enumerate(games):
            game_data = self.df[self.df['Game'] == game_num].sort_values('Round')
            rounds = game_data['Round']
            color = colors[idx]
            
            # Competence Trust
            axes[0].plot(rounds, game_data['S_CTrust'], 
                        marker='o', linewidth=2, label=f'G{game_num} S→R',
                        color=color, alpha=0.8)
            axes[0].plot(rounds, game_data['R_CTrust'], 
                        marker='s', linewidth=2, linestyle='--', label=f'G{game_num} R→S',
                        color=color, alpha=0.6)
            
            # Benevolence Trust
            axes[1].plot(rounds, game_data['S_BTrust'], 
                        marker='o', linewidth=2, label=f'G{game_num} S→R',
                        color=color, alpha=0.8)
            axes[1].plot(rounds, game_data['R_BTrust'], 
                        marker='s', linewidth=2, linestyle='--', label=f'G{game_num} R→S',
                        color=color, alpha=0.6)
        
        axes[0].set_xlabel('Round', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Competence Trust', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Competence Trust - First {max_games} Games', fontsize=13, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)
        
        axes[1].set_xlabel('Round', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Benevolence Trust', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Benevolence Trust - First {max_games} Games', fontsize=13, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_evolution_individual.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_action_distribution(self):
        """Visualize amount sent distribution and trust relationship."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Amount sent distribution by round
        ax1 = axes[0, 0]
        round_data = self.df.groupby('Round')[['Amount Sent By Sender', 'Amount Sent By Receiver']].mean()
        
        x = round_data.index
        width = 0.35
        ax1.bar(x - width/2, round_data['Amount Sent By Sender'], width, 
               label='Sender Amount', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, round_data['Amount Sent By Receiver'], width, 
               label='Receiver Return', color='#A23B72', alpha=0.8)
        
        ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Amount ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Amounts Sent by Round', fontsize=13, fontweight='bold')
        ax1.legend(framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Trust vs Amount sent scatter
        ax2 = axes[0, 1]
        ax2.scatter(self.df['S_BTrust'], self.df['Amount Sent By Sender'], 
                   c='#2E86AB', alpha=0.6, s=50, label='Sender Amount')
        
        # Calculate correlation
        corr_s = self.df['S_BTrust'].corr(self.df['Amount Sent By Sender'])
        ax2.set_xlabel('Sender Benevolence Trust', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amount Sent by Sender ($)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Trust vs Amount Sent (r={corr_s:.2f})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        
        # Plot 3: Return ratio analysis
        ax3 = axes[1, 0]
        self.df['Return_Ratio'] = self.df.apply(
            lambda row: (row['Amount Sent By Receiver'] / (row['Amount Sent By Sender'] * 2) 
                        if row['Amount Sent By Sender'] > 0 else 0), axis=1
        )
        
        trust_bins = pd.cut(self.df['R_BTrust'], bins=[0, 0.3, 0.6, 1.0], 
                           labels=['Low Trust', 'Medium Trust', 'High Trust'])
        return_by_trust = self.df.groupby(trust_bins)['Return_Ratio'].mean()
        
        bars = ax3.bar(return_by_trust.index, return_by_trust.values, 
                      color=['#E63946', '#F4A261', '#2A9D8F'], alpha=0.8)
        ax3.set_ylabel('Average Return Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Receiver Generosity by Trust Level', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Fair Split')
        ax3.legend()
        
        # Plot 4: Wealth distribution
        ax4 = axes[1, 1]
        final_rounds = self.df.groupby('Game').tail(1)
        wealth_data = [final_rounds['Final Sender Amount'], final_rounds['Final Receiver Amount']]
        bp = ax4.boxplot(wealth_data, labels=['Sender', 'Receiver'], 
                        patch_artist=True, notch=True)
        for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_ylabel('Final Amount ($)', fontsize=12, fontweight='bold')
        ax4.set_title('Final Wealth Distribution', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "action_trust_analysis.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_game_outcomes(self):
        """Visualize game outcomes and trust relationship."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Get final game data
        final_rounds = self.df.groupby('Game').tail(1)
        
        # Plot 1: Winner distribution
        ax1 = axes[0]
        winners = []
        for _, row in final_rounds.iterrows():
            if row['Final Sender Amount'] > row['Final Receiver Amount']:
                winners.append('Sender')
            elif row['Final Receiver Amount'] > row['Final Sender Amount']:
                winners.append('Receiver')
            else:
                winners.append('Tie')
        
        winner_counts = pd.Series(winners).value_counts()
        colors_pie = ['#2E86AB', '#A23B72', '#F4A261']
        ax1.pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Game Winners', fontsize=13, fontweight='bold')
        
        # Plot 2: Total wealth by game
        ax2 = axes[1]
        x = np.arange(len(final_rounds))
        width = 0.35
        ax2.bar(x - width/2, final_rounds['Final Sender Amount'], width, 
               label='Sender', color='#2E86AB', alpha=0.8)
        ax2.bar(x + width/2, final_rounds['Final Receiver Amount'], width, 
               label='Receiver', color='#A23B72', alpha=0.8)
        ax2.set_xlabel('Game Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Final Amount ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Final Wealth by Game', fontsize=13, fontweight='bold')
        ax2.legend(framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Trust vs cooperation
        ax3 = axes[2]
        game_stats = []
        for game in self.df['Game'].unique():
            game_data = self.df[self.df['Game'] == game]
            avg_trust = (game_data['S_BTrust'].mean() + game_data['R_BTrust'].mean()) / 2
            avg_sent = game_data['Amount Sent By Sender'].mean()
            total_wealth = game_data.iloc[-1]['Final Sender Amount'] + game_data.iloc[-1]['Final Receiver Amount']
            game_stats.append({'trust': avg_trust, 'cooperation': avg_sent, 'total_wealth': total_wealth})
        
        stats_df = pd.DataFrame(game_stats)
        scatter = ax3.scatter(stats_df['trust'], stats_df['cooperation'], 
                             c=stats_df['total_wealth'], cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Average Game Trust', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Amount Sent', fontsize=12, fontweight='bold')
        ax3.set_title('Trust vs Cooperation', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Total Wealth', fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / "game_outcomes.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_simple_trust_zones(self):
        """Simple plot showing trust zones and what they mean."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Calculate average trust (combine competence and benevolence)
        round_stats = self.df.groupby('Round').agg({
            'S_CTrust': 'mean',
            'S_BTrust': 'mean',
            'R_CTrust': 'mean',
            'R_BTrust': 'mean'
        }).reset_index()
        
        # Average trust for each player
        round_stats['S_Avg_Trust'] = (round_stats['S_CTrust'] + round_stats['S_BTrust']) / 2
        round_stats['R_Avg_Trust'] = (round_stats['R_CTrust'] + round_stats['R_BTrust']) / 2
        
        rounds = round_stats['Round']
        
        # Add colored zones
        ax.axhspan(0, 0.3, alpha=0.15, color='red', label='Low Trust Zone')
        ax.axhspan(0.3, 0.6, alpha=0.15, color='yellow', label='Medium Trust Zone')
        ax.axhspan(0.6, 1.0, alpha=0.15, color='green', label='High Trust Zone')
        
        # Plot trust lines
        ax.plot(rounds, round_stats['S_Avg_Trust'], 
               marker='o', linewidth=3, label='Sender Trust in Receiver', 
               color='#2E86AB', markersize=10)
        ax.plot(rounds, round_stats['R_Avg_Trust'], 
               marker='s', linewidth=3, label='Receiver Trust in Sender', 
               color='#A23B72', markersize=10, linestyle='--')
        
        # Add annotations - positioned to avoid overlap with Y-axis label
        ax.text(1.5, 0.15, 'LOW TRUST\nLikely to send less', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        ax.text(1.5, 0.45, 'MEDIUM TRUST\nModerate amounts', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        ax.text(1.5, 0.8, 'HIGH TRUST\nGenerous sending/returning', 
               ha='center', fontsize=11, fontweight='bold', alpha=0.7)
        
        ax.set_xlabel('Round Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Trust Score', fontsize=14, fontweight='bold')
        ax.set_title('Trust Evolution with Decision Zones\n(Combines Competence + Benevolence Trust)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(rounds.min() - 0.5, rounds.max() + 0.5)
        
        # Format Y-axis for better readability with fewer ticks to avoid overlap
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Format X-axis for better readability
        ax.set_xticks(rounds)
        ax.set_xticklabels([f'{int(r)}' for r in rounds])
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_zones_simple.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_game_summary_cards(self):
        """Create a summary card for each game showing key metrics."""
        games = sorted(self.df['Game'].unique())
        
        # Calculate metrics per game
        game_metrics = []
        for game in games:
            game_data = self.df[self.df['Game'] == game]
            final_row = game_data.iloc[-1]
            
            avg_trust = (game_data['S_CTrust'].mean() + game_data['S_BTrust'].mean() + 
                        game_data['R_CTrust'].mean() + game_data['R_BTrust'].mean()) / 4
            avg_sent = game_data['Amount Sent By Sender'].mean()
            cooperation_rate = (avg_sent / 15) * 100  # Assuming max amount is 15
            
            metrics = {
                'game': game,
                'rounds': game_data['Round'].max(),
                'avg_trust': avg_trust,
                's_payoff': final_row['Final Sender Amount'],
                'r_payoff': final_row['Final Receiver Amount'],
                'cooperation': cooperation_rate,
                'avg_sent': avg_sent
            }
            game_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(game_metrics)
        
        # Create grid of subplots
        n_games = len(games)
        cols = 5
        rows = (n_games + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
        if n_games == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        for idx, (_, row) in enumerate(df_metrics.iterrows()):
            ax = axes[idx]
            
            # Create a simple card layout
            ax.axis('off')
            
            # Background
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2)
            ax.add_patch(rect)
            
            # Title
            ax.text(0.5, 0.85, f'GAME {int(row["game"])}', 
                   ha='center', fontsize=14, fontweight='bold', 
                   transform=ax.transAxes)
            
            # Rounds
            ax.text(0.5, 0.72, f'{int(row["rounds"])} rounds', 
                   ha='center', fontsize=11, style='italic',
                   transform=ax.transAxes)
            
            # Trust scores
            trust_color = '#2ecc71' if row['avg_trust'] > 0.6 else '#f39c12' if row['avg_trust'] > 0.3 else '#e74c3c'
            ax.text(0.5, 0.58, f'Trust: {row["avg_trust"]:.2f}', 
                   ha='center', fontsize=12, color=trust_color, fontweight='bold',
                   transform=ax.transAxes)
            
            # Cooperation rate
            ax.text(0.5, 0.48, f'Avg Sent: ${row["avg_sent"]:.1f}', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            
            # Payoffs
            winner = 'S' if row['s_payoff'] > row['r_payoff'] else 'R' if row['r_payoff'] > row['s_payoff'] else 'TIE'
            winner_color = '#2E86AB' if winner == 'S' else '#A23B72' if winner == 'R' else '#95a5a6'
            
            ax.text(0.5, 0.35, f'Sender: ${int(row["s_payoff"])}', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            ax.text(0.5, 0.25, f'Receiver: ${int(row["r_payoff"])}', 
                   ha='center', fontsize=10,
                   transform=ax.transAxes)
            ax.text(0.5, 0.12, f'Winner: {winner}', 
                   ha='center', fontsize=11, fontweight='bold', color=winner_color,
                   transform=ax.transAxes)
        
        # Hide extra subplots
        for idx in range(n_games, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Game-by-Game Summary Cards', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        output_path = self.output_dir / "game_summary_cards.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_decision_matrix(self):
        """Show when players send high vs low amounts based on trust and round."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Calculate average trust for sender
        self.df['S_Avg_Trust'] = (self.df['S_CTrust'] + self.df['S_BTrust']) / 2
        
        # Define high/low sending based on median
        median_sent = self.df['Amount Sent By Sender'].median()
        self.df['Sending_Level'] = self.df['Amount Sent By Sender'].apply(
            lambda x: 'High Send' if x > median_sent else 'Low Send'
        )
        
        # Plot 1: Trust vs Round colored by sending amount
        ax1 = axes[0]
        low_send = self.df[self.df['Sending_Level'] == 'Low Send']
        high_send = self.df[self.df['Sending_Level'] == 'High Send']
        
        scatter1 = ax1.scatter(high_send['Round'], high_send['S_Avg_Trust'], 
                              c='#2A9D8F', marker='o', s=120, alpha=0.6, 
                              label='High Send (Cooperative)', edgecolors='white', linewidths=1.5)
        scatter2 = ax1.scatter(low_send['Round'], low_send['S_Avg_Trust'], 
                              c='#E63946', marker='X', s=180, alpha=0.8, 
                              label='Low Send (Cautious)', edgecolors='white', linewidths=2)
        
        ax1.set_xlabel('Round Number', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Sender\'s Trust in Receiver', fontsize=13, fontweight='bold')
        ax1.set_title('Sending Patterns: When Do Senders Send More vs Less?', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=12, framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Add zone indicators
        ax1.axhline(0.6, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax1.text(ax1.get_xlim()[1], 0.62, 'High Trust', 
                va='bottom', ha='right', fontsize=10, color='green', fontweight='bold')
        
        # Plot 2: Sending behavior heatmap
        ax2 = axes[1]
        
        # Create bins for trust and round
        self.df['Trust_Bin'] = pd.cut(self.df['S_Avg_Trust'], bins=[0, 0.3, 0.6, 1.0], 
                                       labels=['Low\n(0-0.3)', 'Medium\n(0.3-0.6)', 'High\n(0.6-1.0)'])
        self.df['Round_Bin'] = pd.cut(self.df['Round'], bins=[0, 3, 6, 10], 
                                       labels=['Early\n(1-3)', 'Mid\n(4-6)', 'Late\n(7-10)'])
        
        # Calculate average amount sent in each bin
        heatmap_data = self.df.groupby(['Trust_Bin', 'Round_Bin'])['Amount Sent By Sender'].mean()
        heatmap_data = heatmap_data.unstack()
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   ax=ax2, cbar_kws={'label': 'Average Amount Sent ($)'}, 
                   linewidths=2, linecolor='white')
        ax2.set_xlabel('Game Stage', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Trust Level', fontsize=13, fontweight='bold')
        ax2.set_title('Average Amount Sent by Trust & Game Stage', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "decision_matrix.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_trust_vs_payoff(self):
        """Show relationship between trust building and final payoffs."""
        # Get final game data
        final_games = []
        for game in self.df['Game'].unique():
            game_data = self.df[self.df['Game'] == game]
            final_row = game_data.iloc[-1]
            
            avg_s_trust = (game_data['S_CTrust'].mean() + game_data['S_BTrust'].mean()) / 2
            avg_r_trust = (game_data['R_CTrust'].mean() + game_data['R_BTrust'].mean()) / 2
            avg_trust = (avg_s_trust + avg_r_trust) / 2
            
            final_games.append({
                'game': game,
                'avg_trust': avg_trust,
                's_payoff': final_row['Final Sender Amount'],
                'r_payoff': final_row['Final Receiver Amount'],
                'total_payoff': final_row['Final Sender Amount'] + final_row['Final Receiver Amount'],
                'rounds': game_data['Round'].max(),
                'avg_sent': game_data['Amount Sent By Sender'].mean()
            })
        
        df_final = pd.DataFrame(final_games)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Trust vs Total Payoff
        ax1 = axes[0]
        scatter = ax1.scatter(df_final['avg_trust'], df_final['total_payoff'], 
                             c=df_final['avg_sent'], cmap='viridis', s=200, 
                             alpha=0.7, edgecolors='black', linewidths=2)
        
        # Add trend line
        z = np.polyfit(df_final['avg_trust'], df_final['total_payoff'], 1)
        p = np.poly1d(z)
        ax1.plot(df_final['avg_trust'], p(df_final['avg_trust']), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax1.set_xlabel('Average Trust Level in Game', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Total Payoff (Both Players)', fontsize=13, fontweight='bold')
        ax1.set_title('Does Higher Trust Lead to Higher Payoffs?', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Average Amount Sent', fontsize=11, fontweight='bold')
        
        # Plot 2: Trust development over rounds
        ax2 = axes[1]
        
        # Calculate trust development (final - initial trust)
        trust_changes = []
        for game in self.df['Game'].unique():
            game_data = self.df[self.df['Game'] == game]
            initial_trust = (game_data.iloc[0]['S_CTrust'] + game_data.iloc[0]['S_BTrust'] + 
                           game_data.iloc[0]['R_CTrust'] + game_data.iloc[0]['R_BTrust']) / 4
            final_trust = (game_data.iloc[-1]['S_CTrust'] + game_data.iloc[-1]['S_BTrust'] + 
                          game_data.iloc[-1]['R_CTrust'] + game_data.iloc[-1]['R_BTrust']) / 4
            trust_change = final_trust - initial_trust
            trust_changes.append(trust_change)
        
        colors_change = ['#E63946' if x < -0.1 else '#2A9D8F' if x > 0.1 else '#F4A261' 
                        for x in trust_changes]
        
        bars = ax2.bar(df_final['game'], trust_changes, color=colors_change, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Game Number', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Trust Change (Final - Initial)', fontsize=13, fontweight='bold')
        ax2.set_title('Trust Development Over Games', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#E63946', label='Trust Decreased'),
            Patch(facecolor='#F4A261', label='Trust Stable'),
            Patch(facecolor='#2A9D8F', label='Trust Increased')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "trust_vs_payoff.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_communication_analysis(self):
        """Analyze the relationship between promises and actions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Promise length vs trust
        ax1 = axes[0, 0]
        self.df['Sender_Promise_Length'] = self.df['Sender Promise'].str.len()
        self.df['Receiver_Promise_Length'] = self.df['Receiver Promise'].str.len()
        
        ax1.scatter(self.df['Sender_Promise_Length'], self.df['S_BTrust'], 
                   alpha=0.6, color='#2E86AB', label='Sender')
        ax1.scatter(self.df['Receiver_Promise_Length'], self.df['R_BTrust'], 
                   alpha=0.6, color='#A23B72', label='Receiver')
        
        ax1.set_xlabel('Promise Length (characters)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Benevolence Trust', fontsize=12, fontweight='bold')
        ax1.set_title('Promise Verbosity vs Trust', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trust consistency
        ax2 = axes[0, 1]
        self.df['Trust_Diff'] = abs(self.df['S_BTrust'] - self.df['R_BTrust'])
        round_trust_diff = self.df.groupby('Round')['Trust_Diff'].mean()
        
        ax2.plot(round_trust_diff.index, round_trust_diff.values, 
                marker='o', linewidth=3, color='#E63946', markersize=8)
        ax2.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Trust Difference', fontsize=12, fontweight='bold')
        ax2.set_title('Trust Asymmetry Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reciprocity analysis
        ax3 = axes[1, 0]
        self.df['Expected_Return'] = self.df['Amount Sent By Sender'] * 2 * 0.5  # Fair return
        self.df['Reciprocity'] = self.df['Amount Sent By Receiver'] / self.df['Expected_Return']
        self.df['Reciprocity'] = self.df['Reciprocity'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        reciprocity_by_round = self.df.groupby('Round')['Reciprocity'].mean()
        ax3.bar(reciprocity_by_round.index, reciprocity_by_round.values, 
               color='#2A9D8F', alpha=0.7, edgecolor='black')
        ax3.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Fair Reciprocity')
        ax3.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Reciprocity Ratio', fontsize=12, fontweight='bold')
        ax3.set_title('Receiver Reciprocity by Round', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Promise sentiment (basic)
        ax4 = axes[1, 1]
        # Simple sentiment based on positive words
        positive_words = ['trust', 'fair', 'good', 'great', 'excellent', 'generous', 'kind', 'thank']
        
        def simple_sentiment(text):
            if pd.isna(text):
                return 0
            text_lower = text.lower()
            return sum(1 for word in positive_words if word in text_lower)
        
        self.df['Sender_Sentiment'] = self.df['Sender Promise'].apply(simple_sentiment)
        self.df['Receiver_Sentiment'] = self.df['Receiver Promise'].apply(simple_sentiment)
        
        sentiment_trust_s = self.df.groupby('Sender_Sentiment')['Amount Sent By Sender'].mean()
        sentiment_trust_r = self.df.groupby('Receiver_Sentiment')['Amount Sent By Receiver'].mean()
        
        x_s = sentiment_trust_s.index
        x_r = sentiment_trust_r.index
        
        if len(x_s) > 0:
            ax4.bar(x_s - 0.2, sentiment_trust_s.values, width=0.4, 
                   label='Sender Amount', color='#2E86AB', alpha=0.7)
        if len(x_r) > 0:
            ax4.bar(x_r + 0.2, sentiment_trust_r.values, width=0.4, 
                   label='Receiver Return', color='#A23B72', alpha=0.7)
        
        ax4.set_xlabel('Promise Positivity (word count)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Average Amount ($)', fontsize=12, fontweight='bold')
        ax4.set_title('Promise Sentiment vs Actions', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "communication_analysis.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_interpretation_report(self):
        """Generate text-based interpretation of the results."""
        report = []
        report.append("=" * 70)
        report.append("TRUST GAME ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Basic statistics
        num_games = self.df['Game'].nunique()
        total_rounds = len(self.df)
        avg_game_length = self.df.groupby('Game')['Round'].max().mean()
        
        report.append("1. BASIC STATISTICS")
        report.append(f"   - Total Games: {num_games}")
        report.append(f"   - Total Rounds: {total_rounds}")
        report.append(f"   - Average Game Length: {avg_game_length:.2f} rounds")
        report.append("")
        
        # Trust evolution
        initial_trust = self.df.groupby('Game').first()[['S_CTrust', 'S_BTrust', 'R_CTrust', 'R_BTrust']].mean()
        final_trust = self.df.groupby('Game').last()[['S_CTrust', 'S_BTrust', 'R_CTrust', 'R_BTrust']].mean()
        
        report.append("2. TRUST EVOLUTION")
        report.append(f"   Initial Trust (avg):")
        report.append(f"     - Sender Competence Trust: {initial_trust['S_CTrust']:.3f}")
        report.append(f"     - Sender Benevolence Trust: {initial_trust['S_BTrust']:.3f}")
        report.append(f"     - Receiver Competence Trust: {initial_trust['R_CTrust']:.3f}")
        report.append(f"     - Receiver Benevolence Trust: {initial_trust['R_BTrust']:.3f}")
        report.append(f"   Final Trust (avg):")
        report.append(f"     - Sender Competence Trust: {final_trust['S_CTrust']:.3f}")
        report.append(f"     - Sender Benevolence Trust: {final_trust['S_BTrust']:.3f}")
        report.append(f"     - Receiver Competence Trust: {final_trust['R_CTrust']:.3f}")
        report.append(f"     - Receiver Benevolence Trust: {final_trust['R_BTrust']:.3f}")
        report.append("")
        
        # Amount analysis
        avg_sent = self.df['Amount Sent By Sender'].mean()
        avg_returned = self.df['Amount Sent By Receiver'].mean()
        avg_pot = self.df['Amount Sent By Sender'].mean() * 2
        return_ratio = avg_returned / avg_pot if avg_pot > 0 else 0
        
        report.append("3. MONETARY BEHAVIOR")
        report.append(f"   - Average amount sent by sender: ${avg_sent:.2f}")
        report.append(f"   - Average amount returned by receiver: ${avg_returned:.2f}")
        report.append(f"   - Average return ratio: {return_ratio:.3f} (0.5 = fair split)")
        report.append("")
        
        # Trust-action correlation
        corr_s_trust_amount = self.df['S_BTrust'].corr(self.df['Amount Sent By Sender'])
        corr_r_trust_return = self.df['R_BTrust'].corr(self.df['Amount Sent By Receiver'])
        
        report.append("4. TRUST-ACTION RELATIONSHIPS")
        report.append(f"   - Sender trust vs amount sent correlation: {corr_s_trust_amount:.3f}")
        report.append(f"   - Receiver trust vs amount returned correlation: {corr_r_trust_return:.3f}")
        report.append("")
        
        # Outcomes
        final_outcomes = self.df.groupby('Game').last()[['Final Sender Amount', 'Final Receiver Amount']]
        s_wins = (final_outcomes['Final Sender Amount'] > final_outcomes['Final Receiver Amount']).sum()
        r_wins = (final_outcomes['Final Receiver Amount'] > final_outcomes['Final Sender Amount']).sum()
        ties = (final_outcomes['Final Sender Amount'] == final_outcomes['Final Receiver Amount']).sum()
        
        report.append("5. GAME OUTCOMES")
        report.append(f"   - Sender Wins: {s_wins} ({s_wins/num_games*100:.1f}%)")
        report.append(f"   - Receiver Wins: {r_wins} ({r_wins/num_games*100:.1f}%)")
        report.append(f"   - Ties: {ties} ({ties/num_games*100:.1f}%)")
        report.append(f"   - Average Sender Final Amount: ${final_outcomes['Final Sender Amount'].mean():.2f}")
        report.append(f"   - Average Receiver Final Amount: ${final_outcomes['Final Receiver Amount'].mean():.2f}")
        report.append("")
        
        # Key insights
        report.append("6. KEY INSIGHTS")
        
        if return_ratio > 0.6:
            report.append(f"   ✓ HIGH reciprocity ({return_ratio:.2f}) suggests strong cooperation")
        elif return_ratio > 0.4:
            report.append(f"   → MODERATE reciprocity ({return_ratio:.2f}) indicates cautious cooperation")
        else:
            report.append(f"   ✗ LOW reciprocity ({return_ratio:.2f}) suggests exploitation")
        
        trust_change = (final_trust.mean() - initial_trust.mean()) / initial_trust.mean() * 100
        if trust_change > 10:
            report.append(f"   ✓ Trust INCREASED by {trust_change:.1f}% over games")
        elif trust_change > -10:
            report.append(f"   → Trust remained STABLE (change: {trust_change:.1f}%)")
        else:
            report.append(f"   ✗ Trust DECREASED by {abs(trust_change):.1f}% over games")
        
        if corr_s_trust_amount > 0.5:
            report.append(f"   ✓ Strong trust-action correlation ({corr_s_trust_amount:.2f}) shows rational behavior")
        elif corr_s_trust_amount > 0.2:
            report.append(f"   → Moderate trust-action correlation ({corr_s_trust_amount:.2f})")
        else:
            report.append(f"   ✗ Weak trust-action correlation ({corr_s_trust_amount:.2f}) suggests disconnect")
        
        report.append("")
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\n✓ Report saved: {report_path}")
    
    def run_all_visualizations(self):
        """Run all visualization methods."""
        print("\n" + "="*70)
        print("GENERATING TRUST GAME VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("=== SIMPLE, EASY-TO-UNDERSTAND PLOTS ===")
        print("1. Trust Zones (Simple)...")
        self.plot_simple_trust_zones()
        
        print("2. Game Summary Cards...")
        self.plot_game_summary_cards()
        
        print("3. Decision Matrix (When High vs Low Sending)...")
        self.plot_decision_matrix()
        
        print("4. Trust vs Payoff Analysis...")
        self.plot_trust_vs_payoff()
        
        print("\n=== DETAILED ANALYTICAL PLOTS ===")
        print("5. Trust Evolution (Aggregate)...")
        self.plot_trust_evolution_aggregate()
        
        print("6. Trust Evolution (Individual Games)...")
        self.plot_individual_games(max_games=5)
        
        print("7. Action & Trust Analysis...")
        self.plot_action_distribution()
        
        print("8. Game Outcomes...")
        self.plot_game_outcomes()
        
        print("9. Communication Analysis...")
        self.plot_communication_analysis()
        
        print("\n10. Generating Interpretation Report...")
        self.generate_interpretation_report()
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print(f"✓ All files saved in: {self.output_dir.absolute()}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Find the most recent CSV file
    csv_files = [
        "trust_game_log.csv",
        "sr_games_100m_10r.csv",
        "sr_games_15m_10r.csv"
    ]
    
    csv_path = None
    for file in csv_files:
        if Path(file).exists():
            csv_path = file
            break
    
    if csv_path is None:
        print("❌ No CSV file found! Please run the trust game simulation first.")
    else:
        print(f"\n📊 Using data from: {csv_path}\n")
        visualizer = TrustGameVisualizer(csv_path)
        visualizer.run_all_visualizations()
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("  1. Check the 'trust_plots/' folder for all generated visualizations")
        print("  2. Review 'analysis_report.txt' for detailed interpretation")
        print("\n  START WITH THESE SIMPLE PLOTS:")
        print("     ⭐ trust_zones_simple.png - Trust over time with colored zones")
        print("     ⭐ game_summary_cards.png - One card per game with key stats")
        print("     ⭐ decision_matrix.png - When players send high vs low amounts")
        print("     ⭐ trust_vs_payoff.png - Does trust lead to better outcomes?")
        print("\n  THEN EXPLORE DETAILED PLOTS:")
        print("     - trust_evolution_aggregate.png (statistical averages)")
        print("     - trust_evolution_individual.png (specific game trajectories)")
        print("     - action_trust_analysis.png (comprehensive analysis)")
        print("     - game_outcomes.png (win rates and distributions)")
        print("     - communication_analysis.png (promise vs action analysis)")
        print("="*70 + "\n")
