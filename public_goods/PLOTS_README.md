# Public Goods Game - Visualization Guide

This document describes all the plots generated for analyzing the public goods game results.

## Overview
The plotting script generates **19 comprehensive visualizations** organized into 7 categories, providing insights into player behavior, cooperation patterns, trust dynamics, and game outcomes.

---

## 1. Wealth Evolution Plots

### `wealth_evolution_final_wealth_boxplot.png`
- **Description**: Box plot showing the distribution of final wealth for each player across all games
- **Insights**: 
  - Identifies which players consistently accumulate more wealth
  - Shows wealth variance and outliers for each player
  - Helps identify free-riders vs. cooperators

### `wealth_evolution_average_trajectory.png`
- **Description**: Line plot tracking average wealth evolution for each player over rounds
- **Insights**:
  - Shows wealth growth patterns across the game
  - Reveals when players gain advantages
  - Identifies critical turning points in wealth accumulation

---

## 2. Contribution Pattern Plots

### `contributions_average_by_player.png`
- **Description**: Bar chart of average contributions by each player
- **Insights**:
  - Identifies generous cooperators vs. free-riders
  - Shows baseline cooperation levels
  - Useful for comparing player strategies

### `contributions_rate_evolution.png`
- **Description**: Line plot showing contribution rate (contribution/wealth) over rounds for each player
- **Insights**:
  - Normalizes contributions by wealth to show true cooperation
  - Reveals if players become more/less cooperative over time
  - Shows strategic adaptation patterns

### `contributions_variability.png`
- **Description**: Standard deviation of contributions across all players by round
- **Insights**:
  - High variability suggests diverse strategies
  - Decreasing variability indicates convergence to stable patterns
  - Shows coordination (or lack thereof) among players

---

## 3. Trust Dynamics Plots

### `trust_dynamics.png`
- **Description**: Two-panel plot showing:
  - Left: Evolution of average competence and benevolence trust over rounds
  - Right: Scatter plot of competence vs. benevolence trust with correlation
- **Insights**:
  - Shows how trust builds or erodes over time
  - Reveals relationship between trust types
  - High correlation suggests trust dimensions move together

### `trust_by_player.png`
- **Description**: Line plot of competence trust evolution for each player
- **Insights**:
  - Identifies which players are most trusted
  - Shows trust trajectories over the game
  - Reveals reputation building/destruction

---

## 4. Cooperation Efficiency Plots

### `efficiency_pool_evolution.png`
- **Description**: Average public pool size over rounds
- **Insights**:
  - Shows overall cooperation level
  - Increasing pool suggests growing cooperation
  - Declining pool indicates free-riding or defection

### `efficiency_ratio.png`
- **Description**: Efficiency ratio (distribution/average contribution) over rounds
- **Insights**:
  - Should equal multiplier (2x) with full cooperation
  - Lower values indicate inefficient cooperation
  - Shows whether the group captures multiplier benefits

### `efficiency_net_gain.png`
- **Description**: Average net gain (distribution - contribution) per player by round
- **Insights**:
  - Positive values show net benefit from cooperation
  - Identifies who benefits most from the public pool
  - Reveals free-riding behavior (high gain, low contribution)

---

## 5. Conditional Cooperation Analysis

### `cc_analysis_distribution.png`
- **Description**: Histogram of conditional cooperation (CC) scores
- **Insights**:
  - Distribution around 0 suggests tit-for-tat behavior
  - Positive CC indicates cooperative reciprocity
  - Negative CC shows contrarian or competitive behavior

### `cc_analysis_evolution.png`
- **Description**: Average CC score evolution with confidence bands
- **Insights**:
  - Shows how conditional cooperation changes over time
  - Increasing CC suggests players become more reciprocal
  - Wider bands indicate heterogeneous responses

### `cc_analysis_by_player.png`
- **Description**: Bar chart of average CC scores by player
- **Insights**:
  - Identifies strong reciprocators vs. independent actors
  - Positive CC players match others' contributions
  - Negative CC players may be contrarians or strategic

---

## 6. Trust-Cooperation Relationships (Original Plots)

### `scatter_meancc_ctrust.png`
- **Description**: Scatter plot of mean conditional cooperation vs. mean competence trust
- **Insights**:
  - Shows if competent players are more conditionally cooperative
  - Positive correlation suggests trust enables reciprocity

### `scatter_meancc_btrust.png`
- **Description**: Scatter plot of mean conditional cooperation vs. mean benevolence trust
- **Insights**:
  - Shows if benevolent players reciprocate cooperation
  - Reveals trust-behavior alignment

### `timeseries_game1_player1.png`
- **Description**: Detailed time series for a specific player showing contribution, trust scores, and CC
- **Insights**:
  - Example deep-dive into individual player dynamics
  - Shows how behavior and trust co-evolve
  - Useful for case study analysis

---

## 7. Game Outcome Plots

### `outcomes_winners.png`
- **Description**: Bar chart showing number of wins per player
- **Insights**:
  - Identifies the most successful strategy
  - Shows if outcomes are balanced or dominated

### `outcomes_inequality.png`
- **Description**: Gini coefficient of wealth inequality over rounds
- **Insights**:
  - 0 = perfect equality, 1 = maximum inequality
  - Increasing Gini shows wealth concentration
  - Reveals if cooperation leads to inequality

### `outcomes_rankings.png`
- **Description**: Average ranking position for each player over rounds
- **Insights**:
  - Lower rank = better performance (1 is best)
  - Shows ranking stability vs. volatility
  - Identifies consistent performers

---

## How to Generate Plots

Run the plotting script from the public_goods directory:

```bash
python plots.py
```

### Optional Arguments:
- `--log_csv`: Path to game log CSV (default: `public_goods_game_log.csv`)
- `--cc_csv`: Path to conditional cooperation CSV (default: `cc_metrics.csv`)
- `--game`: Game ID for time series plot (default: 1)
- `--player`: Player ID for time series plot (default: 1)

Example with custom parameters:
```bash
python plots.py --game 5 --player 3
```

---

## Key Findings to Look For

1. **Free-Riding**: Players with high wealth but low contributions
2. **Reciprocity**: Positive CC scores indicating conditional cooperation
3. **Trust Building**: Increasing trust scores correlating with contributions
4. **Strategic Adaptation**: Changes in contribution rates over time
5. **Wealth Inequality**: Gini coefficient trends showing fairness
6. **Cooperation Efficiency**: Pool sizes approaching optimal levels
7. **Winner Patterns**: Which strategies lead to victory

---

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

---

## Notes

- All plots use 300 DPI for publication quality
- NaN values are handled automatically
- Plots are saved to the current directory
- Each plot is designed to be interpretable standalone
