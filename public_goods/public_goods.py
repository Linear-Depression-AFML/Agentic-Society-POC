import math
import random
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  

# --- 1. TRUST TRACKER CLASS ---

class TrustTracker:
    """
    Implements dual-component trust formula for PUBLIC GOODS GAME.
    - CTrust (Competence) is updated by promise quality & action rationality.
    - BTrust (Benevolence) is updated by promise emotion & contribution fairness.
    Trust formula: Tnew = Told + psi/ln(Told + e - 1)
    Where psi can be:
    - Collaborative: psi = Q*(f+D)
    - Information gain: psi = I*R*E
    
    In public goods game:
    - Each agent tracks trust in ALL other agents
    - Actions are evaluated based on contribution to the public pool
    - Fair contribution = equal share (pool_size / num_agents)
    """
    def __init__(self, agent_name, D=0.5, k_c=0.01, k_b=0.01):
        self.agent_name = agent_name
        self.CTrust = 0.5
        self.BTrust = 0.5
        self.e = math.e
        self.D = D
        self.k_c = k_c
        self.k_b = k_b
        # Initialize VADER once for the class
        self.vader_analyzer = SentimentIntensityAnalyzer()
        # Track previous messages for novelty calculation
        self.previous_messages = []
        self.max_message_history = 10

    def _get_promise_effort(self, analyzer_agent, message):
        """
        Calculates Effort (f) based on the level of commitment expressed in the promise.
        Uses LLM to assess the strength of commitment/effort promised.
        Returns a value between 0.0 (no commitment) and 1.0 (strong commitment).
        """
        effort_template = """
You are evaluating the level of commitment/effort expressed in this promise from a PUBLIC GOODS GAME.

Message: '{message}'

Rate the level of commitment and effort implied by this promise on a scale of 0.0 to 1.0:
- 1.0 = Strong commitment (specific contribution promises, concrete cooperation signals)
- 0.5 = Moderate commitment (vague cooperation promises, some effort indicated)
- 0.0 = No commitment (free-riding signals, no contribution intent)

Focus on what they're promising to CONTRIBUTE, not how many words they use.

Examples:
- "I will contribute my full share to maximize our collective benefit" = high effort (0.8-1.0)
- "I'll cooperate fairly with the group" = moderate effort (0.4-0.6)
- "Let's see what others do first" = low effort (0.1-0.3)
- Long rambling with no concrete contribution promise = low effort (0.0-0.2)

Respond with only a single floating-point number from 0.0 to 1.0.
"""
        
        f_chain = ChatPromptTemplate.from_template(effort_template) | analyzer_agent | StrOutputParser()
        f_response = f_chain.invoke({"message": message})
        
        try:
            f = max(0.0, min(1.0, float(f_response.strip())))
        except ValueError:
            f = 0.5  # Default to moderate effort if parsing fails
        
        return f

    def _calculate_information_novelty(self, message):
        """
        Calculates Information Novelty (I) based on how different the message is
        from previous messages.
        Returns a value between 0.0 (completely repetitive) and 1.0 (completely novel).
        """
        if not self.previous_messages:
            return 1.0  # First message is completely novel
        
        # Convert message to set of words (simple bag-of-words approach)
        current_words = set(message.lower().split())
        
        if not current_words:
            return 0.0
        
        # Calculate maximum similarity with any previous message
        max_similarity = 0.0
        for prev_msg in self.previous_messages:
            prev_words = set(prev_msg.lower().split())
            if not prev_words:
                continue
            
            # Jaccard similarity
            intersection = len(current_words & prev_words)
            union = len(current_words | prev_words)
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of similarity
        novelty = 1.0 - max_similarity
        
        # Update message history
        self.previous_messages.append(message)
        if len(self.previous_messages) > self.max_message_history:
            self.previous_messages.pop(0)
        
        return novelty
    
    def _calculate_goal_relevance(self, analyzer_agent, message, context_type="promise"):
        """
        Calculates Goal Relevance (R) - how much the message affects the target's goals.
        Uses LLM to assess relevance to game goals (cooperation, winning, trust-building).
        Returns a value between 0.0 (irrelevant) and 1.0 (highly relevant).
        """
        relevance_template = """
You are evaluating how relevant this message is to the goals in a PUBLIC GOODS GAME.
The main goals are: maximizing collective benefit, encouraging cooperation, and maintaining group trust.

Message: '{message}'
Context: This is a {context_type} in the public goods game.

Rate the relevance of this message to achieving game goals on a scale of 0.0 to 1.0:
- 1.0 = Highly relevant (directly addresses cooperation, contribution strategy, or collective benefit)
- 0.5 = Moderately relevant (mentions goals indirectly)
- 0.0 = Irrelevant (off-topic or empty)

Respond with only a single floating-point number from 0.0 to 1.0.
"""
        
        r_chain = ChatPromptTemplate.from_template(relevance_template) | analyzer_agent | StrOutputParser()
        r_response = r_chain.invoke({"message": message, "context_type": context_type})
        
        try:
            R = max(0.0, min(1.0, float(r_response.strip())))
        except ValueError:
            R = 0.5  # Default to moderate relevance if parsing fails
        
        return R

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

    def _get_deterministic_action_scores(self, contribution, agent_wealth, fair_share):
        """
        Calculates Q_action and E_action (pos/neg) for PUBLIC GOODS GAME.
        Evaluates based on contribution relative to fair share and agent's wealth.
        
        Args:
            contribution: Amount contributed to public pool
            agent_wealth: Agent's total wealth before contribution
            fair_share: Expected fair contribution (typically wealth / num_agents or fixed amount)
        """
        if agent_wealth <= 0: # Avoid division by zero
            return 0.0, 0.0, 0.0 # No action to score

        # --- Calculate Benevolence (Contribution Fairness) ---
        # Compare actual contribution to fair share
        if fair_share == 0:
            benevolence_score = 0.0 if contribution == 0 else 1.0
        else:
            # (0 contributed) -> (0 - fair_share) / fair_share = -1.0 (pure free-riding)
            # (fair_share contributed) -> 0.0 (perfectly fair)
            # (2*fair_share contributed) -> +1.0 (highly altruistic)
            benevolence_score = (contribution - fair_share) / fair_share
            benevolence_score = max(-1.0, min(1.0, benevolence_score))  # Clamp to [-1, 1]
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))

        # --- Calculate Competence (Rationality) ---
        # A rational move contributes enough to benefit from multiplication
        # Completely irrational = pure free-riding (E_neg_action = 1.0)
        Q_action = 1.0 - E_neg_action 
        
        return Q_action, E_pos_action, E_neg_action

    def update_trust_from_promise(self, analyzer_agent, promise):
        """
        Updates trust based *only* on the agent's promise (their text).
        Now includes BOTH collaborative task component AND information gain component.
        """
        self._apply_decay()
        
        if not promise.strip():
            self.CTrust = self.CTrust - 0.1 # Penalize competence
            self.BTrust = self.BTrust - 0.1 # Penalize benevolence
            self._normalize_scores()
            return

        # Get effort based on commitment level, not word count
        f = self._get_promise_effort(analyzer_agent, promise)
        # Get Q from LLM, but E_pos/E_neg from VADER
        Q, E_pos, E_neg = self._get_promise_scores(analyzer_agent, promise)
        
        # Calculate information gain components
        I = self._calculate_information_novelty(promise)  # Information novelty
        R = self._calculate_goal_relevance(analyzer_agent, promise, "promise")  # Goal relevance
        # E_pos is already calculated above (emotional valence from VADER)
        
        # === COLLABORATIVE TASK COMPONENT ===
        # psi_collaborative = Q * (f + D)
        psi_collaborative = Q * (f + self.D)
        
        # === INFORMATION GAIN COMPONENT ===
        # psi_info_gain = I * R * E
        psi_info_gain = I * R * E_pos
        
        # === COMBINE BOTH COMPONENTS FOR COMPETENCE TRUST ===
        # Tnew = Told + psi / ln(Told + e - 1)
        # We combine both psi components
        psi_total_ct = psi_collaborative + psi_info_gain
        ct_bonus = psi_total_ct / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)  # Cap the bonus

        # === BENEVOLENCE TRUST: Positive Emotional Valence ===
        # For benevolence, we use emotional valence (E_pos)
        if E_pos > 0:
            bt_bonus = E_pos / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)  # Cap the bonus
        
        # === NEGATIVE INTERACTIONS: Trust deteriorates ===
        if E_neg > 0:
            self.BTrust = self.BTrust * (1 - E_neg)
            
        self._normalize_scores()

    def update_trust_from_contribution(self, contribution, agent_wealth, fair_share):
        """
        Updates trust based on an agent's CONTRIBUTION to the public pool.
        Includes both collaborative task component AND information gain.
        This is a 100% deterministic calculation for public goods game.
        
        Args:
            contribution: Amount contributed to public pool
            agent_wealth: Agent's total wealth before contribution  
            fair_share: Expected fair contribution
        """
        if agent_wealth <= 0:
            return # Avoid division by zero
        
        # Effort = proportion of wealth contributed
        f_action = contribution / agent_wealth  # 0.0 to 1.0
        
        # Calculate fairness relative to expected contribution
        if fair_share == 0:
            benevolence_score = 0.0 if contribution == 0 else 1.0
        else:
            benevolence_score = (contribution - fair_share) / fair_share
            benevolence_score = max(-1.0, min(1.0, benevolence_score))
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))

        Q_action = 1.0 - E_neg_action
        
        # === COLLABORATIVE TASK COMPONENT ===
        psi_collaborative = Q_action * (f_action + self.D)
        
        # === INFORMATION GAIN FROM ACTION ===
        # Novelty based on deviation from expected fair contribution
        expected_ratio = fair_share / agent_wealth if agent_wealth > 0 else 0.5
        actual_ratio = contribution / agent_wealth
        I_action = abs(actual_ratio - expected_ratio) / max(expected_ratio, 0.1)
        I_action = min(1.0, I_action)  # Cap at 1.0
        
        R_action = 1.0  # Actions are always highly relevant to game goals
        
        psi_info_gain = I_action * R_action * E_pos_action
        
        # === COMBINE BOTH COMPONENTS FOR COMPETENCE TRUST ===
        psi_total = psi_collaborative + psi_info_gain
        ct_bonus = psi_total / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)
        
        # === BENEVOLENCE TRUST: Positive Emotional Valence ===
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        
        # === NEGATIVE INTERACTIONS ===
        if E_neg_action > 0:
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

PLAYER_TEMPLATE = """You are Player {player_id} in a PUBLIC GOODS GAME with {num_players} players.

**GAME RULES:**
- Each player starts with ${initial_wealth} dollars
- Each round, you decide how much to contribute to a public pool (0 to your current wealth)
- The total pool is multiplied by {multiplier}x
- The multiplied amount is distributed EQUALLY among all {num_players} players
- After {total_rounds} rounds, the player with the most money wins

**CURRENT STATUS:**
- Round: {current_round}/{total_rounds}
- Your current wealth: ${your_wealth}
- Other players' wealth: {others_wealth}
- Last round's pool: ${last_pool} (each player received ${last_distribution})

**YOUR TRUST IN OTHER PLAYERS:**
{trust_scores}

**MEMORY OF RECENT ROUNDS:**
{memory}

**STRATEGIC CONSIDERATIONS:**
- If everyone contributes fairly, everyone benefits from the multiplier
- Free-riders benefit from others' contributions without contributing
- Your contribution affects your trust scores from other players
- Trust influences future cooperation

Make a strategic decision based on your trust scores, the game state, and your goal to maximize your final wealth.

Your response MUST be formatted as:
1. A single paragraph explaining your reasoning and strategy
2. On a new line: "Contribution: X" where X is the amount (0 to {your_wealth})

Example:
Given the cooperative behavior in previous rounds, I will contribute a fair share to encourage continued cooperation and benefit from the multiplier effect.
Contribution: 7
"""

# --- Analyzer Prompts (ONLY FOR TEXT QUALITY) ---

Q_PROMISE_TEMPLATE = """
You are an objective AI evaluator. Score the *quality* of this statement in a PUBLIC GOODS GAME.
A high-quality statement (1.0) is rational, clear, and fosters group cooperation.
A low-quality statement (0.0) is irrational, vague, or promotes free-riding.
The STATEMENT is: '{message}'.
Respond with only a single floating-point number from 0.0 to 1.0.
"""

# --- 3. GAME LOGIC ---

def parse_output(output):
    """Extracts the statement and contribution amount from the agent's output."""
    try:
        lines = output.strip().split('\n')
        statement = "\n".join(lines[:-1]).strip()
        contribution_line = lines[-1]
        match = re.search(r'\d+', contribution_line)
        if match:
            contribution = int(match.group(0))
            return statement, contribution
        else:
            return statement, 0
    except Exception as e:
        print(f"[Parsing Error: {e} | Output: {output}]")
        return output, 0

def run_round(players, player_wealth, round_num, total_rounds, player_memories, player_trust_trackers, 
              multiplier, initial_wealth, last_pool, last_distribution):
    """
    Runs one round of the public goods game.
    
    Args:
        players: List of player IDs
        player_wealth: Dict mapping player_id -> current wealth
        round_num: Current round number
        total_rounds: Total number of rounds in game
        player_memories: Dict mapping player_id -> list of memory strings
        player_trust_trackers: Dict mapping (player_i, player_j) -> TrustTracker
        multiplier: Pool multiplier
        initial_wealth: Starting wealth for each player
        last_pool: Total pool from last round
        last_distribution: Amount distributed to each player last round
        
    Returns:
        contributions: Dict mapping player_id -> contribution amount
        statements: Dict mapping player_id -> statement text
        updated_memories: Updated player_memories dict
    """
    num_players = len(players)
    contributions = {}
    statements = {}
    player_chain = ChatPromptTemplate.from_template(PLAYER_TEMPLATE) | mistral | StrOutputParser()
    
    # Each player makes their decision
    for player_id in players:
        # Format other players' wealth
        others_wealth_str = ", ".join([f"P{pid}: ${player_wealth[pid]}" 
                                       for pid in players if pid != player_id])
        
        # Format trust scores for this player
        trust_scores_str = ""
        for other_id in players:
            if other_id != player_id:
                ct, bt = player_trust_trackers[(player_id, other_id)].get_scores()
                trust_scores_str += f"  - Player {other_id}: Competence={ct:.2f}, Benevolence={bt:.2f}\n"
        
        # Get recent memory
        memory_str = "\n".join(player_memories[player_id][-3:]) if player_memories[player_id] else "No previous rounds"
        
        # Invoke the LLM
        output = player_chain.invoke({
            "player_id": player_id,
            "num_players": num_players,
            "initial_wealth": initial_wealth,
            "multiplier": multiplier,
            "current_round": round_num + 1,
            "total_rounds": total_rounds,
            "your_wealth": player_wealth[player_id],
            "others_wealth": others_wealth_str,
            "last_pool": last_pool,
            "last_distribution": last_distribution,
            "trust_scores": trust_scores_str,
            "memory": memory_str
        })
        
        statement, contribution = parse_output(output)
        contribution = max(0, min(player_wealth[player_id], contribution))
        
        contributions[player_id] = contribution
        statements[player_id] = statement
        
        # Update memory
        player_memories[player_id].append(
            f"Round {round_num + 1}: Contributed ${contribution}. Statement: {statement[:50]}..."
        )
    
    return contributions, statements, player_memories

def run_game(initial_wealth, num_players, multiplier, game_number, df, rounds, env_params):
    """
    Runs a full public goods game.
    
    Args:
        initial_wealth: Starting wealth for each player
        num_players: Number of players in the game
        multiplier: Factor by which the public pool is multiplied
        game_number: Game ID for logging
        df: DataFrame for logging results
        rounds: Number of rounds to play
        env_params: Environment parameters for trust calculation
    """
    players = list(range(1, num_players + 1))
    player_wealth = {pid: initial_wealth for pid in players}
    player_memories = {pid: [] for pid in players}
    
    # Create trust trackers: each player tracks trust in all other players
    player_trust_trackers = {}
    for i in players:
        for j in players:
            if i != j:
                player_trust_trackers[(i, j)] = TrustTracker(f"P{i}_trust_in_P{j}", **env_params)
    
    # Analyzer for statement quality
    analyzer_agent = OllamaLLM(model="llama3.2", temperature=0.0)
    
    print(f"\n=== Game {game_number} ===")
    print(f"Players: {num_players}, Initial wealth: ${initial_wealth}, Multiplier: {multiplier}x")
    
    last_pool = 0
    last_distribution = 0
    
    for round_num in range(rounds):
        print(f"\n--- Round {round_num + 1}/{rounds} ---")
        print(f"Player wealth: {player_wealth}")
        
        # Run the round - players make contributions
        contributions, statements, player_memories = run_round(
            players, player_wealth, round_num, rounds, player_memories, 
            player_trust_trackers, multiplier, initial_wealth, last_pool, last_distribution
        )
        
        # Calculate public pool and distribution
        total_pool = sum(contributions.values())
        multiplied_pool = total_pool * multiplier
        distribution_per_player = multiplied_pool / num_players
        
        print(f"\nContributions: {contributions}")
        print(f"Total pool: ${total_pool} -> Multiplied: ${multiplied_pool}")
        print(f"Each player receives: ${distribution_per_player:.2f}")
        
        # Update wealth
        for pid in players:
            player_wealth[pid] = player_wealth[pid] - contributions[pid] + distribution_per_player
        
        # Calculate fair share for trust updates
        # Fair share = equal contribution that would maximize group benefit
        fair_share = player_wealth[pid] * 0.5  # Simple heuristic: contribute half
        
        # --- TRUST UPDATES ---
        print(f"\n  Updating trust scores...")
        
        # 1. Each player updates trust based on statements
        for i in players:
            for j in players:
                if i != j:
                    player_trust_trackers[(i, j)].update_trust_from_promise(
                        analyzer_agent, statements[j]
                    )
        
        # 2. Each player updates trust based on contributions
        for i in players:
            for j in players:
                if i != j:
                    # Fair share is relative - we use average contribution as baseline
                    avg_contribution = total_pool / num_players
                    player_trust_trackers[(i, j)].update_trust_from_contribution(
                        contributions[j], 
                        player_wealth[j] + contributions[j] - distribution_per_player,  # wealth before contribution
                        avg_contribution
                    )
        
        # Log results
        for pid in players:
            # Calculate average trust this player has in others
            trust_scores = [player_trust_trackers[(pid, other)].get_scores() 
                           for other in players if other != pid]
            avg_ctrust = sum(ct for ct, bt in trust_scores) / len(trust_scores) if trust_scores else 0.5
            avg_btrust = sum(bt for ct, bt in trust_scores) / len(trust_scores) if trust_scores else 0.5
            
            new_row = {
                "Game": game_number,
                "Round": round_num + 1,
                "Player": pid,
                "Wealth_Start": player_wealth[pid] - distribution_per_player + contributions[pid],
                "Contribution": contributions[pid],
                "Wealth_End": player_wealth[pid],
                "Statement": statements[pid][:100],  # Truncate for CSV
                "Total_Pool": total_pool,
                "Distribution": distribution_per_player,
                "Avg_CTrust": avg_ctrust,
                "Avg_BTrust": avg_btrust
            }
            df.loc[len(df)] = new_row
        
        last_pool = total_pool
        last_distribution = distribution_per_player
        
        print(f"\nRound {round_num + 1} complete. Updated wealth: {player_wealth}")
    
    # Determine winner
    winner = max(player_wealth, key=player_wealth.get)
    print(f"\n=== Game {game_number} Complete ===")
    print(f"Final wealth: {player_wealth}")
    print(f"Winner: Player {winner} with ${player_wealth[winner]:.2f}")
    
    return winner, df

if __name__ == "__main__":
    # --- 4. SIMULATION PARAMETERS ---
    
    # Environment Parameters
    env_params = {
        'D': 0.5,   # Task Difficulty (0.0-1.0)
        'k_c': 0.01, # Competence Decay Rate
        'k_b': 0.02  # Benevolence Decay Rate
    }
    
    # Public Goods Game Parameters
    NUM_GAMES = 10
    NUM_ROUNDS = 10
    NUM_PLAYERS = 5  # Number of players in each game
    INITIAL_WEALTH = 20  # Starting wealth for each player
    MULTIPLIER = 2.0  # Pool multiplier (common values: 1.5-2.5)

    # Initialize DataFrame
    columns = [
        "Game", "Round", "Player", "Wealth_Start", "Contribution",
        "Wealth_End", "Statement", "Total_Pool", "Distribution",
        "Avg_CTrust", "Avg_BTrust"
    ]
    df = pd.DataFrame(columns=columns)
    
    # Track winners
    winners = {pid: 0 for pid in range(1, NUM_PLAYERS + 1)}
    game_fails = 0

    for i in range(NUM_GAMES):
        try:
            winner, df = run_game(
                INITIAL_WEALTH, NUM_PLAYERS, MULTIPLIER, 
                i + 1, df, NUM_ROUNDS, env_params
            )
            winners[winner] += 1
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            game_fails += 1
    
    df.to_csv("public_goods_game_log.csv", index=False)
    print("\n" + "="*50)
    print(f"Simulation complete. Results saved to public_goods_game_log.csv")
    print(f"Winner distribution: {winners}")
    print(f"Failed games: {game_fails}")
    print("="*50)