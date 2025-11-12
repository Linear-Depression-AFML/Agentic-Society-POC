import math
import random
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- PUBLIC GOODS GAME CONFIGURATION ---
# In this game, N agents each receive an endowment and can contribute any amount to a common pool.
# The pool is multiplied by a factor M and then distributed equally among all agents.
# Individual optimal: contribute 0 (free-ride)
# Social optimal: everyone contributes everything

NUM_AGENTS = 5  # Number of players in the game
MULTIPLIER = 2.0  # Pool multiplier (must be > 1 and < NUM_AGENTS for social dilemma)

# --- 1. TRUST TRACKER CLASS ---

class TrustTracker:
    """
    Implements dual-component trust formula for Public Goods Game.
    - CTrust (Competence) is updated by promise quality & contribution rationality.
    - BTrust (Benevolence) is updated by promise emotion & contribution fairness.
    
    In Public Goods Game:
    - Rational contribution = balancing self-interest with cooperation
    - Fair contribution = contributing proportionally or more to the public good
    """
    def __init__(self, observer_name, target_name, D=0.5, k_c=0.01, k_b=0.01):
        self.observer_name = observer_name  # Who is tracking trust
        self.target_name = target_name      # Who is being trusted
        self.CTrust = 0.5
        self.BTrust = 0.5
        self.e = math.e
        self.D = D
        self.k_c = k_c
        self.k_b = k_b
        # Initialize VADER once for the class
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def _get_f(self, message):
        """Calculates Effort (f) as a deterministic proxy (word count)."""
        word_count = len(message.split())
        return min(1.0, word_count / 50.0) # Normalize, 1.0 = 50 words

    def _apply_decay(self):
        """Applies the constant time decay to both scores."""
        self.CTrust = max(0.01, self.CTrust - self.k_c)
        self.BTrust = max(0.01, self.BTrust - self.k_b)

    def _get_promise_scores(self, analyzer_agent, promise):
        """
        Gets Q (Quality) using the LLM Analyzer.
        Gets E (Emotion) using the deterministic VADER tool.
        """
        # --- 1. Get Quality (Q) from LLM Analyzer ---
        q_chain = ChatPromptTemplate.from_template(Q_PROMISE_TEMPLATE) | analyzer_agent | StrOutputParser()
        q_response = q_chain.invoke({"message": promise})
        try:
            Q = max(0.0, min(1.0, float(q_response.strip())))
        except ValueError:
            Q = 0.0 

        # --- 2. Get Emotion (E_pos, E_neg) from VADER ---
        vader_scores = self.vader_analyzer.polarity_scores(promise)
        compound_score = vader_scores['compound'] # This is the -1.0 to +1.0 score
        
        E_pos = max(0, compound_score)
        E_neg = abs(min(0, compound_score))
            
        return Q, E_pos, E_neg


    def _get_contribution_scores(self, contribution, endowment, avg_contribution):
        """
        Calculates Q_action and E_action (pos/neg) for a contribution decision.
        
        In Public Goods Game:
        - Fair contribution = contributing at or above the average
        - Rational contribution = not free-riding completely (showing some cooperation)
        - Selfish = contributing significantly below average (free-riding)
        """
        if endowment <= 0:
            return 0.0, 0.0, 0.0
        
        # --- Calculate Benevolence (Cooperation vs Free-riding) ---
        # Compare to average contribution
        if avg_contribution == 0:
            # If no one else contributed, any contribution is generous
            benevolence_score = contribution / endowment if endowment > 0 else 0.0
        else:
            # (contribution - avg) / avg
            # Contributing more than average = positive
            # Contributing less than average = negative (free-riding)
            benevolence_score = (contribution - avg_contribution) / avg_contribution
        
        # Normalize to [-1, 1] range
        benevolence_score = max(-1.0, min(1.0, benevolence_score))
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))
        
        # --- Calculate Competence (Strategic Rationality) ---
        # Contributing nothing is purely selfish (low rationality in cooperative context)
        # Contributing some amount shows strategic cooperation
        contribution_ratio = contribution / endowment
        Q_action = contribution_ratio  # Higher contribution = more cooperative rationality
        
        return Q_action, E_pos_action, E_neg_action

    def update_trust_from_promise(self, analyzer_agent, promise):
        """Updates trust based *only* on the agent's promise (their text)."""
        self._apply_decay()
        
        if not promise.strip():
            self.CTrust = self.CTrust - 0.1 # Penalize competence
            self.BTrust = self.BTrust - 0.1 # Penalize benevolence
            self._normalize_scores()
            return

        f = self._get_f(promise)
        # Get Q from LLM, but E_pos/E_neg from VADER
        Q, E_pos, E_neg = self._get_promise_scores(analyzer_agent, promise) 
        
        # Promise Quality updates Competence Trust
        ct_bonus = (Q * (f + self.D)) / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus) # Cap the bonus

        # Promise Emotion updates Benevolence Trust
        if E_pos > 0:
            bt_bonus = E_pos / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus) # Cap the bonus
        if E_neg > 0:
            self.BTrust = self.BTrust * (1 - E_neg)
            
        self._normalize_scores()
    
    def update_trust_from_contribution(self, contribution, endowment, avg_contribution):
        """
        Updates trust based on an agent's contribution to the public good.
        This is a deterministic calculation based on contribution behavior.
        """
        self._apply_decay()
        
        f = 0.5  # Effort for contribution decision
        
        Q_action, E_pos_action, E_neg_action = self._get_contribution_scores(
            contribution, endowment, avg_contribution
        )
        
        # Strategic cooperation updates Competence Trust
        ct_bonus = (Q_action * (f + self.D)) / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)
        
        # Fair/Selfish contribution updates Benevolence Trust
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        if E_neg_action > 0:
            # Penalize free-riding behavior
            self.BTrust = self.BTrust * (1 - E_neg_action)
        
        self._normalize_scores()


    def _normalize_scores(self):
        """Keeps scores between 0.01 and 1.0."""
        self.CTrust = max(0.01, min(1.0, self.CTrust))
        self.BTrust = max(0.01, min(1.0, self.BTrust))

    def get_scores(self):
        return self.CTrust, self.BTrust

# --- 2. AGENT AND PROMPT CONFIGURATION ---

mistral = OllamaLLM(model="mistral")

AGENT_TEMPLATE = """You are Agent {agent_id} in a Public Goods Game with {num_agents} players.

**GAME RULES:**
- Each player starts with ${endowment}
- You can contribute any amount (0 to ${endowment}) to a common pool
- The total pool will be multiplied by {multiplier}x and distributed EQUALLY among all {num_agents} players
- Your final money = (endowment - your contribution) + (total pool × {multiplier} / {num_agents})
- Goal: Maximize your final money over {rounds} rounds

**CURRENT GAME STATE (Round {current_round}):**
- Your current total: ${your_money}
{other_agents_info}

**YOUR TRUST IN OTHER AGENTS:**
{trust_info}

**RECENT HISTORY:**
{memory}

**STRATEGIC CONSIDERATIONS:**
- If everyone contributes everything: each gets ${endowment} × {multiplier} / {num_agents} = ${theoretical_max}
- If only you contribute ${endowment} and others contribute 0: you get ${worst_case}
- If you contribute 0 and others contribute: you free-ride on their cooperation

Based on your trust in others, the game state, and your goal, make a strategic decision.
Your response MUST be a single paragraph explaining your reasoning, followed by your contribution on a new line.

Example:
Given the trust levels and past behavior, I believe moderate cooperation is optimal.
Contribution: 7
"""

# --- Analyzer Prompts (ONLY FOR TEXT QUALITY) ---

Q_PROMISE_TEMPLATE = """
You are an objective AI evaluator. Score the *quality* of this reasoning in a public goods game.
A high-quality reasoning (1.0) is rational, strategic, clear, and considers cooperation-defection tradeoffs.
A low-quality reasoning (0.0) is irrational, vague, or ignores the strategic context.
The REASONING is: '{message}'.
Respond with only a single floating-point number from 0.0 to 1.0.
"""

# --- 3. GAME LOGIC ---


def parse_output(output):
    """Extracts the reasoning and the contribution amount from the agent's output."""
    try:
        lines = output.strip().split('\n')
        reasoning = "\n".join(lines[:-1]).strip()
        contribution_line = lines[-1]
        match = re.search(r'\d+', contribution_line)
        if match:
            contribution = int(match.group(0))
            return reasoning, contribution
        else:
            return reasoning, 0
    except Exception as e:
        print(f"[Parsing Error: {e} | Output: {output}]")
        return output, 0

def run_round(endowment, rounds_left, round_num, agent_money, agent_memories, trust_trackers):
    """
    Run one round of the public goods game with all agents.
    Returns: contributions, reasonings, updated memories
    """
    agent_chain = ChatPromptTemplate.from_template(AGENT_TEMPLATE) | mistral | StrOutputParser()
    
    contributions = []
    reasonings = []
    
    for agent_id in range(NUM_AGENTS):
        # Build trust info for this agent
        trust_lines = []
        for other_id in range(NUM_AGENTS):
            if other_id != agent_id:
                ct, bt = trust_trackers[agent_id][other_id].get_scores()
                trust_lines.append(f"  - Agent {other_id}: Competence={ct:.2f}, Benevolence={bt:.2f}")
        trust_info = "\n".join(trust_lines) if trust_lines else "  (No other agents)"
        
        # Build other agents info
        other_info_lines = []
        for other_id in range(NUM_AGENTS):
            if other_id != agent_id:
                other_info_lines.append(f"  - Agent {other_id}: ${agent_money[other_id]}")
        other_agents_info = "\n".join(other_info_lines)
        
        # Calculate theoretical values for the prompt
        theoretical_max = int(endowment * MULTIPLIER / NUM_AGENTS)
        worst_case = 0  # If only you contribute and get it back divided
        
        # Get agent's memory
        memory_text = "\n".join(agent_memories[agent_id][-3:]) if agent_memories[agent_id] else "No history yet"
        
        # Invoke the agent
        output = agent_chain.invoke({
            "agent_id": agent_id,
            "num_agents": NUM_AGENTS,
            "endowment": endowment,
            "multiplier": MULTIPLIER,
            "rounds": rounds_left,
            "current_round": round_num,
            "your_money": agent_money[agent_id],
            "other_agents_info": other_agents_info,
            "trust_info": trust_info,
            "memory": memory_text,
            "theoretical_max": theoretical_max,
            "worst_case": worst_case
        })
        
        reasoning, contribution = parse_output(output)
        contribution = max(0, min(endowment, contribution))
        
        contributions.append(contribution)
        reasonings.append(reasoning)
        
        # Update memory
        agent_memories[agent_id].append(f"Round {round_num}: Contributed {contribution}")
    
    return contributions, reasonings, agent_memories

def run_game(endowment, game_number, df, rounds, env_params):
    """
    Run a complete public goods game with multiple agents over multiple rounds.
    """
    # Initialize agent money
    agent_money = [endowment] * NUM_AGENTS
    agent_memories = [[] for _ in range(NUM_AGENTS)]
    
    # Analyzer for reasoning quality
    analyzer_agent = OllamaLLM(model="mistral", temperature=0.0)
    
    # Trust trackers: trust_trackers[i][j] = Agent i's trust in Agent j
    trust_trackers = []
    for i in range(NUM_AGENTS):
        trust_trackers.append({})
        for j in range(NUM_AGENTS):
            if i != j:
                trust_trackers[i][j] = TrustTracker(f"Agent{i}", f"Agent{j}", **env_params)
    
    print(f"\n=== Game {game_number} ===")
    
    for round_num in range(1, rounds + 1):
        print(f"\nRound {round_num}:")
        print(f"Agent money: {agent_money}")
        
        # Get all agents' decisions
        contributions, reasonings, agent_memories = run_round(
            endowment, rounds - round_num, round_num, 
            agent_money, agent_memories, trust_trackers
        )
        
        # Calculate the public pool
        total_pool = sum(contributions)
        multiplied_pool = total_pool * MULTIPLIER
        distribution = multiplied_pool / NUM_AGENTS
        
        print(f"Contributions: {contributions}")
        print(f"Total pool: {total_pool} → Multiplied: {multiplied_pool} → Each gets: {distribution:.2f}")
        
        # Calculate average contribution for trust updates
        avg_contribution = total_pool / NUM_AGENTS
        
        # --- TRUST UPDATES ---
        print(f"  Analyzing reasonings and contributions...")
        
        for observer_id in range(NUM_AGENTS):
            for target_id in range(NUM_AGENTS):
                if observer_id != target_id:
                    # Update trust based on target's reasoning
                    trust_trackers[observer_id][target_id].update_trust_from_promise(
                        analyzer_agent, reasonings[target_id]
                    )
                    
                    # Update trust based on target's contribution
                    trust_trackers[observer_id][target_id].update_trust_from_contribution(
                        contributions[target_id], endowment, avg_contribution
                    )
        
        # Update agent money
        for agent_id in range(NUM_AGENTS):
            agent_money[agent_id] = agent_money[agent_id] - contributions[agent_id] + distribution
        
        print(f"Updated agent money: {agent_money}")
        
        # Log to dataframe
        for agent_id in range(NUM_AGENTS):
            # Get average trust this agent has in others
            avg_ct = sum(trust_trackers[agent_id][j].get_scores()[0] 
                        for j in range(NUM_AGENTS) if j != agent_id) / (NUM_AGENTS - 1)
            avg_bt = sum(trust_trackers[agent_id][j].get_scores()[1] 
                        for j in range(NUM_AGENTS) if j != agent_id) / (NUM_AGENTS - 1)
            
            new_row = {
                "Game": game_number,
                "Round": round_num,
                "Agent": agent_id,
                "Contribution": contributions[agent_id],
                "Final_Money": agent_money[agent_id],
                "Avg_CTrust": avg_ct,
                "Avg_BTrust": avg_bt,
                "Reasoning": reasonings[agent_id].strip()
            }
            df.loc[len(df)] = new_row
    
    # Determine winner
    max_money = max(agent_money)
    winners = [i for i, m in enumerate(agent_money) if m == max_money]
    print(f"\nGame {game_number} complete. Winner(s): Agent {winners} with ${max_money:.2f}")
    
    return winners[0] if len(winners) == 1 else -1, df  # Return -1 for ties



if __name__ == "__main__":
    # --- 4. SIMULATION PARAMETERS ---
    
    # Environment Parameters
    env_params = {
        'D': 0.5,   # Task Difficulty (0.0-1.0)
        'k_c': 0.01, # Competence Decay Rate
        'k_b': 0.02  # Benevolence Decay Rate
    }
    
    # Simulation Parameters
    NUM_GAMES = 10
    NUM_ROUNDS = 10
    ENDOWMENT = 10  # Each agent starts with this per round

    # Initialize DataFrame
    columns = [
        "Game", "Round", "Agent", "Contribution", "Final_Money",
        "Avg_CTrust", "Avg_BTrust", "Reasoning"
    ]
    df = pd.DataFrame(columns=columns)
    
    agent_wins = [0] * NUM_AGENTS
    ties = 0
    game_fails = 0

    for i in range(NUM_GAMES):
        try:
            winner, df = run_game(ENDOWMENT, i + 1, df, NUM_ROUNDS, env_params)
            df.to_csv("public_goods_game_log.csv", index=False)
            if winner >= 0:
                agent_wins[winner] += 1
                print(f"Game {i+1} winner: Agent {winner}")
            else:
                ties += 1
                print(f"Game {i+1}: Tie")
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            game_fails += 1
    
    df.to_csv("public_goods_game_log.csv", index=False)
    print("\n" + "="*50)
    print(f"Simulation complete. Results saved to public_goods_game_log.csv")
    print(f"Agent wins: {agent_wins}")
    print(f"Ties: {ties}, Fails: {game_fails}")
    print("="*50)