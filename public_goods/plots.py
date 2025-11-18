# plots_public_goods.py

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    make_scatter_aggregated(df_log, df_cc, prefix=args.scatter_prefix)
    make_timeseries_plot(df_log, df_cc, args.game, args.player, out_prefix=args.timeseries_prefix)


if __name__ == "__main__":
    main()
