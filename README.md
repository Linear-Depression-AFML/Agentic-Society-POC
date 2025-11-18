# Trusting an LLM: Behavioral Game Theory with AI Agents

This repository documents an empirical study exploring strategic interaction, cooperation, and emergent non-rational behaviors in classic economic games, utilizing Large Language Models (LLMs) as autonomous agents. We test how LLM agents deviate from subgame perfect equilibrium predictions and investigate the role of trust, reciprocity, and communication in fostering cooperation.

-----

## Authors

| Name | ID |
| :--- | :--- |
| **Prateek P** | PES1UG23AM211 |
| **Noel Jose** | PES1UG23AM197 |
| **Nihal RG** | PES1UG23AM187 |
| **Rithwik Adwaith** | PES1UG23AM238 |

-----

## Project Overview

The core objective of this project is to simulate human-like decision-making in strategic settings by deploying LLMs as players in well-established game theory experiments. By analyzing the agents' actions and communication patterns across repeated rounds, we aim to uncover **behavioral phenomena** that challenge traditional, purely rational economic models.

### Key Behavioral Phenomena Investigated:

1.  **Deviation from Nash Equilibrium:** Observing systematic non-rational behavior (e.g., non-minimal offers in the Ultimatum Game, and over-cooperation in the Centipede Game).
2.  **Loss Aversion Panic:** Examining instances where agents, when falling behind, choose **risk-seeking strategies** (e.g., greedy, high-variance offers) to try and "break even," directly violating the rational prediction of maximizing expected utility.
3.  **Strategic Fairness:** Analyzing how LLM agents use **moral framing** and communication to legitimize self-serving actions, mirroring human tendencies to weaponize "fairness" to gain a competitive advantage.
4.  **Trust Dynamics:** Quantifying the evolution and breakdown of trust in repeated-interaction games.

-----

## Simulated Game Environments

The repository contains implementations and analysis for several key games:

| Game | Folder | Focus |
| :--- | :--- | :--- |
| **Ultimatum Game** | `ultimatum_game/` | Proposer's offer strategies and Responder's willingness to **punish unfairness**. |
| **Centipede Game** | `centipede_game/` | Trust and cooperation under the **threat of back-induction** (rational defection). |
| **Public Goods Game** | `public_goods/` | **Collective action problems**, free-riding, and the evolution of group cooperation. |
| **Repeated Sender-Receiver Game** | `Sender_Reciever_Game/` | The impact of **cheap talk** (unverifiable communication) and reputation on trust. |

-----

## Repository Structure

The core codebase and analysis are organized by game:

```
Trusting-an-LLM/
├── Sender_Reciever_Game/
│   ├── 1v1_comms/             # Repeated game with explicit LLM communication
│   ├── 1v1_no_comms/          # Repeated game without communication (Baseline)
│   └── ...
├── centipede_game/
│   ├── centipede_game.py      # Core simulation logic
│   ├── visualize_results.py   # Plotting scripts
│   └── plots/                 # Output charts (trust evolution, payoffs)
├── public_goods/
│   ├── public_goods.py        # Core simulation logic
│   ├── plots.py               # Analysis and visualization scripts
│   └── ...
└── ultimatum_game/
    ├── ultimatum_game.py      # Core simulation logic
    ├── ultimatum_analysis.py  # Analysis of offers and rejections
    └── Plots/                 # Output charts (offer distribution, strategic analysis)
```

-----

## Getting Started

### Prerequisites

  * Python (3.9+)
  * Standard scientific libraries (e.g., `pandas`, `numpy`, `matplotlib`, `seaborn`)
  * LLM API access (The specific LLM used for the agents is configured within the game files, often requiring an API key).

### Running Simulations and Analysis

1.  **Clone the Repository:**
    ```bash
    git clone [repository-url]
    cd Trusting-an-LLM
    ```
2.  **Install Dependencies:**
    ```bash
    # Assuming a requirements.txt file exists or using a base setup:
    pip install pandas numpy matplotlib seaborn
    # You will also need the relevant library for the LLM API (e.g., openai, google-genai)
    ```
3.  **Configure LLM Agent:**
      * Set up your LLM API key as an environment variable (e.g., `export OPENAI_API_KEY='your-key'`).
      * Modify the relevant agent class within each game's `_game.py` file to specify the model and parameters.
4.  **Execute a Simulation:**
      * Navigate to the game directory (e.g., `cd ultimatum_game/`).
      * Run the simulation script:
        ```bash
        python ultimatum_game.py
        ```
5.  **Visualize Results:**
      * Run the analysis script to generate plots:
        ```bash
        python ultimatum_analysis.py
        ```
