# plots_public_goods.py

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_logs_and_cc(log_path: str, cc_path: str):
    df_log = pd.read_csv(log_path)
    df_cc = pd.read_csv(cc_path)

    # sanity
    for c in ["Game", "Player", "Round"]:
        if c not in df_log.columns:
            raise ValueError(f"{c} missing in {log_path}")
        if c not in df_cc.columns:
            raise ValueError(f"{c} missing in {cc_path}")
    return df_log, df_cc


def make_scatter_aggregated(df_log: pd.DataFrame, df_cc: pd.DataFrame, prefix: str = "scatter"):
    """
    Scatter plots:
    - Mean_CC vs Avg_CTrust (mean over rounds per game-player)
    - Mean_CC vs Avg_BTrust
    """
    trust_agg = df_log.groupby(["Game", "Player"]).agg({
        "Avg_CTrust": "mean",
        "Avg_BTrust": "mean"
    }).reset_index()

    cc_agg = df_cc.groupby(["Game", "Player"])["CC"].mean().reset_index()
    cc_agg = cc_agg.rename(columns={"CC": "Mean_CC"})

    merged = pd.merge(trust_agg, cc_agg, on=["Game", "Player"], how="inner").dropna(subset=["Mean_CC"])

    if merged.empty:
        print("[WARN] No overlapping data for aggregated scatter plots.")
        return

    # Scatter: Mean_CC vs Avg_CTrust
    plt.figure()
    plt.scatter(merged["Mean_CC"], merged["Avg_CTrust"])
    plt.xlabel("Mean Conditional Cooperation (Mean_CC)")
    plt.ylabel("Mean Avg_CTrust")
    plt.title("Mean_CC vs Mean Avg_CTrust (per Game-Player)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_meancc_ctrust.png", dpi=300)
    print(f"[INFO] Saved {prefix}_meancc_ctrust.png")

    # Scatter: Mean_CC vs Avg_BTrust
    plt.figure()
    plt.scatter(merged["Mean_CC"], merged["Avg_BTrust"])
    plt.xlabel("Mean Conditional Cooperation (Mean_CC)")
    plt.ylabel("Mean Avg_BTrust")
    plt.title("Mean_CC vs Mean Avg_BTrust (per Game-Player)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_meancc_btrust.png", dpi=300)
    print(f"[INFO] Saved {prefix}_meancc_btrust.png")


def make_timeseries_plot(df_log: pd.DataFrame,
                         df_cc: pd.DataFrame,
                         game: int,
                         player: int,
                         out_prefix: str = "timeseries"):
    """
    Plot over rounds for a single (Game, Player):
    - Contribution
    - Avg_CTrust
    - Avg_BTrust
    - CC
    """
    log_sub = df_log[(df_log["Game"] == game) & (df_log["Player"] == player)].copy()
    cc_sub = df_cc[(df_cc["Game"] == game) & (df_cc["Player"] == player)].copy()

    if log_sub.empty:
        print(f"[WARN] No log data for Game={game}, Player={player}")
        return

    # Merge CC info into log so we have all in one
    merged = pd.merge(
        log_sub,
        cc_sub[["Game", "Player", "Round", "CC"]],
        on=["Game", "Player", "Round"],
        how="left"
    ).sort_values("Round")

    rounds = merged["Round"].values
    contrib = merged["Contribution"].values
    ctrust = merged["Avg_CTrust"].values
    btrust = merged["Avg_BTrust"].values
    cc_vals = merged["CC"].values

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Contribution")
    ax1.plot(rounds, contrib, marker="o", label="Contribution", linestyle="--")
    ax1.tick_params(axis="y")

    # Second axis for trust + CC
    ax2 = ax1.twinx()
    ax2.set_ylabel("Trust / CC")

    # Trust lines
    ax2.plot(rounds, ctrust, marker="s", linestyle="--", label="Avg_CTrust")
    ax2.plot(rounds, btrust, marker="^", linestyle="--", label="Avg_BTrust")

    # CC may not exist for the first round, so it will show NaN
    ax2.plot(rounds, cc_vals, marker="x", linestyle="-.", label="CC")

    # Title and legend handling
    fig.suptitle(f"Game {game}, Player {player}: Contribution, Trust, CC over Rounds")
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    out_path = f"{out_prefix}_game{game}_player{player}.png"
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved {out_path}")


def plot_wealth_evolution(df_log: pd.DataFrame, prefix: str = "wealth_evolution"):
    """
    Plot how wealth evolves over rounds for each player across all games.
    Shows the distribution of final wealth and wealth trajectories.
    """
    # Plot 1: Final wealth distribution
    plt.figure(figsize=(10, 6))
    final_wealth = df_log.groupby(['Game', 'Player'])['Wealth_End'].last().reset_index()
    
    sns.boxplot(data=final_wealth, x='Player', y='Wealth_End')
    plt.title('Final Wealth Distribution by Player')
    plt.xlabel('Player')
    plt.ylabel('Final Wealth ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_final_wealth_boxplot.png", dpi=300)
    print(f"[INFO] Saved {prefix}_final_wealth_boxplot.png")
    
    # Plot 2: Average wealth evolution across all games
    plt.figure(figsize=(12, 6))
    avg_wealth = df_log.groupby(['Round', 'Player'])['Wealth_End'].mean().reset_index()
    
    for player in sorted(avg_wealth['Player'].unique()):
        player_data = avg_wealth[avg_wealth['Player'] == player]
        plt.plot(player_data['Round'], player_data['Wealth_End'], marker='o', label=f'Player {player}')
    
    plt.title('Average Wealth Evolution Across All Games')
    plt.xlabel('Round')
    plt.ylabel('Average Wealth ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_average_trajectory.png", dpi=300)
    print(f"[INFO] Saved {prefix}_average_trajectory.png")
    
    plt.close('all')


def plot_contribution_patterns(df_log: pd.DataFrame, prefix: str = "contributions"):
    """
    Analyze contribution patterns: average contributions, contribution rates,
    and how contributions change over time.
    """
    # Plot 1: Average contribution by player
    plt.figure(figsize=(10, 6))
    avg_contrib = df_log.groupby('Player')['Contribution'].mean().reset_index()
    
    plt.bar(avg_contrib['Player'], avg_contrib['Contribution'])
    plt.title('Average Contribution by Player Across All Games')
    plt.xlabel('Player')
    plt.ylabel('Average Contribution ($)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{prefix}_average_by_player.png", dpi=300)
    print(f"[INFO] Saved {prefix}_average_by_player.png")
    
    # Plot 2: Contribution rate over rounds
    plt.figure(figsize=(12, 6))
    df_log['Contribution_Rate'] = df_log['Contribution'] / df_log['Wealth_Start']
    avg_rate = df_log.groupby(['Round', 'Player'])['Contribution_Rate'].mean().reset_index()
    
    for player in sorted(avg_rate['Player'].unique()):
        player_data = avg_rate[avg_rate['Player'] == player]
        plt.plot(player_data['Round'], player_data['Contribution_Rate'], marker='o', label=f'Player {player}')
    
    plt.title('Average Contribution Rate Evolution (Contribution/Wealth)')
    plt.xlabel('Round')
    plt.ylabel('Contribution Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_rate_evolution.png", dpi=300)
    print(f"[INFO] Saved {prefix}_rate_evolution.png")
    
    # Plot 3: Contribution variance by round
    plt.figure(figsize=(12, 6))
    contrib_std = df_log.groupby('Round')['Contribution'].std().reset_index()
    
    plt.plot(contrib_std['Round'], contrib_std['Contribution'], marker='o', color='red', linewidth=2)
    plt.title('Contribution Variability Across Rounds')
    plt.xlabel('Round')
    plt.ylabel('Standard Deviation of Contributions ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_variability.png", dpi=300)
    print(f"[INFO] Saved {prefix}_variability.png")
    
    plt.close('all')


def plot_trust_dynamics(df_log: pd.DataFrame, df_cc: pd.DataFrame, prefix: str = "trust"):
    """
    Visualize trust dynamics: trust evolution, correlation between trust types,
    and relationship between trust and cooperation.
    """
    # Plot 1: Trust evolution over rounds
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    avg_ctrust = df_log.groupby('Round')['Avg_CTrust'].mean()
    avg_btrust = df_log.groupby('Round')['Avg_BTrust'].mean()
    
    ax1.plot(avg_ctrust.index, avg_ctrust.values, marker='o', label='Competence Trust', linewidth=2)
    ax1.plot(avg_btrust.index, avg_btrust.values, marker='s', label='Benevolence Trust', linewidth=2)
    ax1.set_title('Average Trust Evolution Over Rounds')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Trust Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation between trust types
    ax2.scatter(df_log['Avg_CTrust'], df_log['Avg_BTrust'], alpha=0.5)
    ax2.set_title('Competence vs Benevolence Trust')
    ax2.set_xlabel('Competence Trust')
    ax2.set_ylabel('Benevolence Trust')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = df_log[['Avg_CTrust', 'Avg_BTrust']].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{prefix}_dynamics.png", dpi=300)
    print(f"[INFO] Saved {prefix}_dynamics.png")
    
    # Plot 3: Trust by player over time
    plt.figure(figsize=(14, 6))
    
    for player in sorted(df_log['Player'].unique()):
        player_data = df_log[df_log['Player'] == player].groupby('Round')[['Avg_CTrust', 'Avg_BTrust']].mean()
        plt.plot(player_data.index, player_data['Avg_CTrust'], marker='o', linestyle='--', 
                label=f'P{player} CTrust', alpha=0.7)
    
    plt.title('Competence Trust Evolution by Player')
    plt.xlabel('Round')
    plt.ylabel('Average Competence Trust')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_by_player.png", dpi=300)
    print(f"[INFO] Saved {prefix}_by_player.png")
    
    plt.close('all')


def plot_cooperation_efficiency(df_log: pd.DataFrame, prefix: str = "efficiency"):
    """
    Analyze cooperation efficiency: total pool size, distribution efficiency,
    and collective benefit over time.
    """
    # Plot 1: Total pool evolution
    plt.figure(figsize=(12, 6))
    avg_pool = df_log.groupby('Round')['Total_Pool'].mean()
    
    plt.plot(avg_pool.index, avg_pool.values, marker='o', linewidth=2, color='green')
    plt.title('Average Public Pool Size Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Average Pool Size ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_pool_evolution.png", dpi=300)
    print(f"[INFO] Saved {prefix}_pool_evolution.png")
    
    # Plot 2: Efficiency metric (Distribution per player vs average contribution)
    plt.figure(figsize=(12, 6))
    df_log['Efficiency'] = df_log['Distribution'] / (df_log['Total_Pool'] / 5)
    avg_efficiency = df_log.groupby('Round')['Efficiency'].mean()
    
    plt.plot(avg_efficiency.index, avg_efficiency.values, marker='o', linewidth=2, color='purple')
    plt.axhline(y=2.0, color='r', linestyle='--', label='Multiplier (2x)')
    plt.title('Cooperation Efficiency Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Efficiency Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_ratio.png", dpi=300)
    print(f"[INFO] Saved {prefix}_ratio.png")
    
    # Plot 3: Collective wealth gain per round
    plt.figure(figsize=(12, 6))
    df_log['Net_Gain'] = df_log['Distribution'] - df_log['Contribution']
    avg_net_gain = df_log.groupby(['Round', 'Player'])['Net_Gain'].mean().reset_index()
    
    for player in sorted(avg_net_gain['Player'].unique()):
        player_data = avg_net_gain[avg_net_gain['Player'] == player]
        plt.plot(player_data['Round'], player_data['Net_Gain'], marker='o', label=f'Player {player}')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Average Net Gain per Round by Player (Distribution - Contribution)')
    plt.xlabel('Round')
    plt.ylabel('Net Gain ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_net_gain.png", dpi=300)
    print(f"[INFO] Saved {prefix}_net_gain.png")
    
    plt.close('all')


def plot_conditional_cooperation_analysis(df_cc: pd.DataFrame, prefix: str = "cc_analysis"):
    """
    Analyze conditional cooperation patterns and their distribution.
    """
    # Remove NaN values
    df_cc_clean = df_cc.dropna(subset=['CC'])
    
    if df_cc_clean.empty:
        print("[WARN] No valid CC data for analysis")
        return
    
    # Plot 1: CC distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_cc_clean['CC'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='CC = 0')
    plt.title('Distribution of Conditional Cooperation Scores')
    plt.xlabel('Conditional Cooperation (CC)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{prefix}_distribution.png", dpi=300)
    print(f"[INFO] Saved {prefix}_distribution.png")
    
    # Plot 2: CC evolution over rounds
    plt.figure(figsize=(12, 6))
    avg_cc = df_cc_clean.groupby('Round')['CC'].mean()
    std_cc = df_cc_clean.groupby('Round')['CC'].std()
    
    plt.plot(avg_cc.index, avg_cc.values, marker='o', linewidth=2, color='blue')
    plt.fill_between(avg_cc.index, avg_cc.values - std_cc.values, 
                     avg_cc.values + std_cc.values, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Conditional Cooperation Evolution Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Average CC Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_evolution.png", dpi=300)
    print(f"[INFO] Saved {prefix}_evolution.png")
    
    # Plot 3: CC by player
    plt.figure(figsize=(10, 6))
    avg_cc_player = df_cc_clean.groupby('Player')['CC'].mean().reset_index()
    
    plt.bar(avg_cc_player['Player'], avg_cc_player['CC'])
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Average Conditional Cooperation by Player')
    plt.xlabel('Player')
    plt.ylabel('Average CC Score')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{prefix}_by_player.png", dpi=300)
    print(f"[INFO] Saved {prefix}_by_player.png")
    
    plt.close('all')


def plot_game_outcomes(df_log: pd.DataFrame, prefix: str = "outcomes"):
    """
    Analyze game outcomes: winner distribution, final rankings, wealth inequality.
    """
    # Get final wealth for each game
    final_data = df_log[df_log['Round'] == df_log.groupby(['Game', 'Player'])['Round'].transform('max')]
    
    # Plot 1: Winner distribution
    plt.figure(figsize=(10, 6))
    winners = final_data.loc[final_data.groupby('Game')['Wealth_End'].idxmax()]
    winner_counts = winners['Player'].value_counts().sort_index()
    
    plt.bar(winner_counts.index, winner_counts.values)
    plt.title('Number of Games Won by Each Player')
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{prefix}_winners.png", dpi=300)
    print(f"[INFO] Saved {prefix}_winners.png")
    
    # Plot 2: Wealth inequality (Gini coefficient over rounds)
    plt.figure(figsize=(12, 6))
    
    def gini(x):
        # Calculate Gini coefficient
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n
    
    gini_by_round = df_log.groupby('Round')['Wealth_End'].apply(gini)
    
    plt.plot(gini_by_round.index, gini_by_round.values, marker='o', linewidth=2, color='orange')
    plt.title('Wealth Inequality Over Rounds (Gini Coefficient)')
    plt.xlabel('Round')
    plt.ylabel('Gini Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_inequality.png", dpi=300)
    print(f"[INFO] Saved {prefix}_inequality.png")
    
    # Plot 3: Average ranking consistency
    plt.figure(figsize=(12, 6))
    df_log['Rank'] = df_log.groupby(['Game', 'Round'])['Wealth_End'].rank(ascending=False)
    avg_rank = df_log.groupby(['Round', 'Player'])['Rank'].mean().reset_index()
    
    for player in sorted(avg_rank['Player'].unique()):
        player_data = avg_rank[avg_rank['Player'] == player]
        plt.plot(player_data['Round'], player_data['Rank'], marker='o', label=f'Player {player}')
    
    plt.title('Average Ranking by Player Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Average Rank (1 = Best)')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_rankings.png", dpi=300)
    print(f"[INFO] Saved {prefix}_rankings.png")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Plot CC vs Trust for Public Goods Game.")
    parser.add_argument("--log_csv", type=str, default="public_goods_game_log.csv",
                        help="Path to public_goods_game_log.csv")
    parser.add_argument("--cc_csv", type=str, default="cc_metrics.csv",
                        help="Path to cc_metrics.csv from metrics_public_goods.py")
    parser.add_argument("--scatter_prefix", type=str, default="scatter",
                        help="Prefix for scatter plot filenames")
    parser.add_argument("--timeseries_prefix", type=str, default="timeseries",
                        help="Prefix for timeseries plot filenames")
    parser.add_argument("--game", type=int, default=1,
                        help="Game ID to visualize in time series")
    parser.add_argument("--player", type=int, default=1,
                        help="Player ID to visualize in time series")

    args = parser.parse_args()

    df_log, df_cc = load_logs_and_cc(args.log_csv, args.cc_csv)
    
    # Original plots
    make_scatter_aggregated(df_log, df_cc, prefix=args.scatter_prefix)
    make_timeseries_plot(df_log, df_cc, args.game, args.player, out_prefix=args.timeseries_prefix)
    
    # New comprehensive plots
    print("\n[INFO] Generating wealth evolution plots...")
    plot_wealth_evolution(df_log)
    
    print("\n[INFO] Generating contribution pattern plots...")
    plot_contribution_patterns(df_log)
    
    print("\n[INFO] Generating trust dynamics plots...")
    plot_trust_dynamics(df_log, df_cc)
    
    print("\n[INFO] Generating cooperation efficiency plots...")
    plot_cooperation_efficiency(df_log)
    
    print("\n[INFO] Generating conditional cooperation analysis plots...")
    plot_conditional_cooperation_analysis(df_cc)
    
    print("\n[INFO] Generating game outcome plots...")
    plot_game_outcomes(df_log)
    
    print("\n[INFO] All plots generated successfully!")


if __name__ == "__main__":
    main()
