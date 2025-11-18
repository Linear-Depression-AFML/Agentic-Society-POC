# metrics_public_goods.py

import pandas as pd
import numpy as np
import argparse


def load_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "Game", "Round", "Player",
        "Contribution", "Avg_CTrust", "Avg_BTrust"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Ensure sorted
    df = df.sort_values(by=["Game", "Player", "Round"]).reset_index(drop=True)
    return df


def compute_avg_others_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Game, Round, Player), compute Avg_Others_Contribution:
    = (sum of contributions of others) / (num_players - 1)
    """
    # group by game+round to compute total and count
    grp = df.groupby(["Game", "Round"])["Contribution"]
    total_per_round = grp.transform("sum")
    count_per_round = grp.transform("count")

    df["Avg_Others_Contribution"] = (total_per_round - df["Contribution"]) / (count_per_round - 1)
    return df


def compute_cc_per_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conditional cooperation (CC) per (Game, Player, Round).

    Using:
        CC_i at round t = Δc_i / Δ(Avg_Others_Contribution)
        where Δ is difference between current and previous round for that player.

    This attaches CC to the *current* round index (i.e., from second round onward).
    """
    records = []

    for (game, player), group in df.groupby(["Game", "Player"]):
        group = group.sort_values("Round")

        contrib = group["Contribution"].values
        others_avg = group["Avg_Others_Contribution"].values
        rounds = group["Round"].values

        # differences (length n-1)
        delta_self = np.diff(contrib)
        delta_others = np.diff(others_avg)

        for i in range(1, len(rounds)):
            r = rounds[i]
            ds = delta_self[i - 1]
            do = delta_others[i - 1]

            if abs(do) < 1e-9:
                cc = np.nan
            else:
                cc = ds / do

            records.append({
                "Game": game,
                "Player": player,
                "Round": r,
                "CC": cc
            })

    cc_df = pd.DataFrame(records)
    return cc_df


def merge_cc_with_trust(df_log: pd.DataFrame, cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CC with trust metrics at the same (Game, Player, Round).
    Only rounds >= 2 will have CC values (since it needs a diff).
    """
    merged = pd.merge(
        cc_df,
        df_log[["Game", "Player", "Round", "Avg_CTrust", "Avg_BTrust"]],
        on=["Game", "Player", "Round"],
        how="left"
    )
    return merged


def compute_correlations(merged_round_level: pd.DataFrame,
                         df_log: pd.DataFrame,
                         cc_df: pd.DataFrame) -> None:
    """
    Prints:
    - Round-level correlation: CC vs Avg_CTrust / Avg_BTrust
    - Player-level aggregated correlation: mean CC vs mean Avg_CTrust / Avg_BTrust
    """
    # --- Round-level correlations ---
    round_data = merged_round_level.dropna(subset=["CC", "Avg_CTrust", "Avg_BTrust"]).copy()

    if not round_data.empty:
        corr_ctrust_round = round_data["CC"].corr(round_data["Avg_CTrust"])
        corr_btrust_round = round_data["CC"].corr(round_data["Avg_BTrust"])
    else:
        corr_ctrust_round = np.nan
        corr_btrust_round = np.nan

    # --- Player-level aggregated correlations ---
    # Aggregate trust per (Game, Player)
    trust_agg = df_log.groupby(["Game", "Player"]).agg({
        "Avg_CTrust": "mean",
        "Avg_BTrust": "mean"
    }).reset_index()

    # Aggregate CC per (Game, Player)
    cc_agg = cc_df.groupby(["Game", "Player"])["CC"].mean().reset_index()
    cc_agg = cc_agg.rename(columns={"CC": "Mean_CC"})

    agg_merged = pd.merge(trust_agg, cc_agg, on=["Game", "Player"], how="left")
    agg_merged = agg_merged.dropna(subset=["Mean_CC"])

    if not agg_merged.empty:
        corr_ctrust_player = agg_merged["Mean_CC"].corr(agg_merged["Avg_CTrust"])
        corr_btrust_player = agg_merged["Mean_CC"].corr(agg_merged["Avg_BTrust"])
    else:
        corr_ctrust_player = np.nan
        corr_btrust_player = np.nan

    print("\n===== ROUND-LEVEL CORRELATIONS =====")
    print(f"CC vs Avg_CTrust  (round-level): {corr_ctrust_round:.4f}")
    print(f"CC vs Avg_BTrust  (round-level): {corr_btrust_round:.4f}")

    print("\n===== PLAYER-LEVEL (AGGREGATED) CORRELATIONS =====")
    print(f"Mean CC vs Mean Avg_CTrust (per player/game): {corr_ctrust_player:.4f}")
    print(f"Mean CC vs Mean Avg_BTrust (per player/game): {corr_btrust_player:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compute CC metric and correlations for Public Goods Game logs.")
    parser.add_argument(
        "--csv",
        type=str,
        default="public_goods_game_log.csv",
        help="Path to public_goods_game_log.csv"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cc_metrics.csv",
        help="Output CSV path for CC + trust per round"
    )
    args = parser.parse_args()

    df_log = load_log(args.csv)
    df_log = compute_avg_others_contribution(df_log)
    cc_df = compute_cc_per_round(df_log)
    merged_round = merge_cc_with_trust(df_log, cc_df)

    # Save per-round CC + trust
    merged_round.to_csv(args.out, index=False)
    print(f"[INFO] Saved per-round CC + trust to: {args.out}")

    # Print correlations
    compute_correlations(merged_round, df_log, cc_df)


if __name__ == "__main__":
    main()
