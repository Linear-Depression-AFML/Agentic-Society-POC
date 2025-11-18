import math
import random
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # <-- VADER IMPORT

# --- 1. TRUST TRACKER CLASS (MODIFIED FOR ULTIMATUM GAME) ---

class TrustTracker:
    """
    Implements your dual-component trust formula.
    - CTrust (Competence) is updated by promise quality & action rationality.
    - BTrust (Benevolence) is updated by promise emotion & action fairness.
    Trust formula: Tnew = Told + psi/ln(Told + e - 1)
    Where psi can be:
    - Collaborative: psi = Q*(f+D)
    - Information gain: psi = I*R*E
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
You are evaluating the level of commitment/effort expressed in this promise from a trust game.

Message: '{message}'

Rate the level of commitment and effort implied by this promise on a scale of 0.0 to 1.0:
- 1.0 = Strong commitment (specific promises, concrete actions, high investment indicated)
- 0.5 = Moderate commitment (vague promises, some effort indicated)
- 0.0 = No commitment (empty words, no effort, just talk)

Focus on what they're promising to DO, not how many words they use.

Examples:
- "I will send you 10 dollars" = high effort (0.8-1.0)
- "I trust you and will be fair" = moderate effort (0.4-0.6)
- "Let's see what happens" = low effort (0.1-0.3)
- Long rambling with no concrete promise = low effort (0.0-0.2)

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
You are evaluating how relevant this message is to the goals in a trust game.
The main goals are: maximizing money, building cooperation, and maintaining trust.

Message: '{message}'
Context: This is a {context_type} in the trust game.

Rate the relevance of this message to achieving game goals on a scale of 0.0 to 1.0:
- 1.0 = Highly relevant (directly addresses cooperation, strategy, or trust)
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

    def _get_proposer_offer_scores(self, pot_size, amount_offered):
        """
        Calculates Q_action and E_action (pos/neg) for the Proposer's offer.
        This is a deterministic calculation of fairness.
        """
        if pot_size <= 0: # Avoid division by zero
            return 0.0, 0.0, 0.0 # No action to score

        # --- Calculate Benevolence (Fairness) ---
        fair_offer = pot_size / 2.0
        
        if fair_offer == 0:
            benevolence_score = 0.0 if amount_offered == 0 else 1.0
        else:
            # (0 offered) -> (0 - 10) / 10 = -1.0 (purely selfish)
            # (10 offered) -> (10 - 10) / 10 = 0.0 (perfectly fair)
            # (20 offered) -> (20 - 10) / 10 = +1.0 (purely altruistic)
            benevolence_score = (amount_offered - fair_offer) / fair_offer
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))

        # --- Calculate Competence (Rationality) ---
        # A rational move (Q=1.0) is any move that is not highly selfish
        # (which would guarantee rejection)
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

    def update_trust_from_action_proposer(self, pot_size, amount_offered):
        """
        Updates the *Responder's* trust in the *Proposer* based on the offer.
        This is 100% deterministic and uses both trust components.
        """
        if pot_size <= 0:
            return # No action to score
            
        Q_action, E_pos_action, E_neg_action = self._get_proposer_offer_scores(
            pot_size, amount_offered
        )
        
        # Effort for proposer = proportion of pot they're offering (risk/generosity)
        f_action = amount_offered / pot_size
        
        # === COLLABORATIVE TASK COMPONENT ===
        psi_collaborative = Q_action * (f_action + self.D)
        
        # === INFORMATION GAIN FROM ACTION ===
        # Calculate novelty based on deviation from expected fair offer
        expected_ratio = 0.5  # Fair split
        actual_ratio = amount_offered / pot_size
        I_action = abs(actual_ratio - expected_ratio) / 0.5  # Normalize deviation
        I_action = min(1.0, I_action)  # Cap at 1.0
        
        R_action = 1.0  # Actions are always highly relevant to game goals
        
        psi_info_gain = I_action * R_action * E_pos_action
        
        # === COMBINE BOTH COMPONENTS FOR COMPETENCE TRUST ===
        psi_total = psi_collaborative + psi_info_gain
        ct_bonus = psi_total / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)

        # === BENEVOLENCE TRUST: Positive/Negative Emotional Valence ===
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        if E_neg_action > 0:
            # This will trigger on selfish offers
            self.BTrust = self.BTrust * (1 - E_neg_action)
            
        self._normalize_scores()
        
    def update_trust_from_action_responder(self, pot_size, amount_offered, decision):
        """
        Updates the *Proposer's* trust in the *Responder* based on their decision.
        decision = 1 (Accept), 0 (Reject)
        Uses both trust components.
        """
        f = 0.5 # Effort for a binary action is fixed
        Q_action = 0.0
        E_pos_action = 0.0
        E_neg_action = 0.0
        
        # --- Calculate Competence (Rationality) ---
        if amount_offered > 0:
            Q_action = 1.0 if decision == 1 else 0.0 # Accepting > 0 is rational
        else:
            Q_action = 1.0 # If offer is 0, both decisions have same payoff ($0)
            
        # --- Calculate Benevolence (Fairness/Spite) ---
        is_offer_unfair = amount_offered < (pot_size * 0.3) # e.g., < $6 on a $20 pot
        
        if decision == 1: # Accept
            E_pos_action = 0.1 # Accepting is a neutral/cooperative act
        else: # Reject
            if is_offer_unfair:
                # "Altruistic punishment" - rejecting a bad offer is pro-social
                E_pos_action = 0.8 
            else:
                # "Spite" - rejecting a fair offer is purely negative
                E_neg_action = 1.0 

        # === COLLABORATIVE TASK COMPONENT ===
        psi_collaborative = Q_action * (f + self.D)
        
        # === INFORMATION GAIN FROM ACTION ===
        R_action = 1.0 # Decision is always relevant
        
        # Novelty is based on how unexpected the decision is
        I_action = 0.0
        if decision == 1: # Accept
            I_action = 1.0 if is_offer_unfair else 0.0 # Accepting unfair offer is novel
        else: # Reject
            I_action = 1.0 if not is_offer_unfair else 0.0 # Rejecting fair offer is novel
            
        psi_info_gain = I_action * R_action * E_pos_action
        
        # === COMBINE BOTH COMPONENTS FOR COMPETENCE TRUST ===
        psi_total = psi_collaborative + psi_info_gain
        
        # Update CTrust
        ct_bonus = psi_total / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)
        
        # Update BTrust
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        if E_neg_action > 0:
            self.BTrust = self.BTrust * (1 - E_neg_action)
            
        self._normalize_scores()

    def _normalize_scores(self):
        """Keeps scores between 0.01 and 1.0."""
        self.CTrust = max(0.01, min(1.0, self.CTrust))
        self.BTrust = max(0.01, min(1.0, self.BTrust))

    def get_scores(self):
        return self.CTrust, self.BTrust

# --- 2. AGENT AND PROMPT CONFIGURATION (NEW) ---

mistral = OllamaLLM(model="mistral")

PROPOSER_TEMPLATE = """You are the {role}. You are given a pot of ${pot_size} to split.
You must make an offer to the Responder (from 0 to {pot_size}).
If the Responder accepts, you get (${pot_size} - your offer) and they get your offer.
If the Responder rejects, you *both get $0*.
At the end of the game, the person with the most total money wins.
There are {rounds} rounds left.
Your total score: ${proposer_acc}. Responder's total score: ${responder_acc}.

**YOUR TRUST IN THE RESPONDER:**
(Scores from 0.0 to 1.0)
- **Competence Trust (Rationality): {p_ctrust:.2f}**
- **Benevolence Trust (Fairness/Spite): {p_btrust:.2f}**

Based on your trust, your goal, and the game state, make a strategic decision.
Your response MUST be a single paragraph of your promise, followed by the amount you OFFER on a new line.
Example:
I think this is a fair deal for both of us.
Amount: 8
"""

RESPONDER_TEMPLATE = """You are the {role}. The Proposer has offered you ${amount_offered} from a total pot of ${pot_size}.
You must choose to **Accept** or **Reject** this offer.
If you accept, you get ${amount_offered} and the Proposer gets ${proposer_gets}.
If you reject, you *both get $0*.
At the end of the game, the person with the most total money wins.
There are {rounds} rounds left.
Your total score: ${responder_acc}. Proposer's total score: ${proposer_acc}.

**YOUR TRUST IN THE PROPOSER:**
(Scores from 0.0 to 1.0)
- **Competence Trust (Rationality): {r_ctrust:.2f}**
- **Benevolence Trust (Fairness): {r_btrust:.2f}**

Based on your trust, your goal, and the game state, make a strategic decision.
Your response MUST be a single paragraph of your promise, followed by 'Decision: Accept' or 'Decision: Reject' on a new line.
Example:
That seems reasonable.
Decision: Accept
"""

# --- Analyzer Prompts (ONLY FOR TEXT QUALITY) ---

Q_PROMISE_TEMPLATE = """
You are an objective AI evaluator. Score the *quality* of this promise in an investment game.
A high-quality promise (1.0) is rational, clear, and fosters cooperation.
A low-quality promise (0.0) is irrational, vague, or purely selfish.
The PROMISE is: '{message}'.
Respond with only a single floating-point number from 0.0 to 1.0.
"""

# --- 3. GAME LOGIC (NEW) ---

def parse_output_proposer(output, pot_size):
    """Extracts the promise and the amount offered."""
    try:
        lines = output.strip().split('\n')
        promise = "\n".join(lines[:-1]).strip()
        amount_line = lines[-1]
        match = re.search(r'\d+', amount_line)
        if match:
            amount = int(match.group(0))
            return promise, max(0, min(pot_size, amount)) # Clamp to pot size
        else:
            return promise, 0
    except Exception as e:
        print(f"[Parsing Error (Proposer): {e} | Output: {output}]")
        return output, 0

def parse_output_responder(output):
    """Extracts the promise and the decision (1 for Accept, 0 for Reject)."""
    try:
        lines = output.strip().split('\n')
        promise = "\n".join(lines[:-1]).strip()
        decision_line = lines[-1].strip().lower()
        
        if "accept" in decision_line:
            return promise, 1
        elif "reject" in decision_line:
            return promise, 0
        else:
            print(f"[Parsing Warning (Responder): No clear decision. Defaulting to Reject. | Output: {output}]")
            return promise, 0 # Default to Reject
    except Exception as e:
        print(f"[Parsing Error (Responder): {e} | Output: {output}]")
        return output, 0

def run_round(pot_size, rounds_left, proposer_acc, responder_acc, p_trust_scores, r_trust_scores):
    proposer_chain = ChatPromptTemplate.from_template(PROPOSER_TEMPLATE) | mistral | StrOutputParser()
    responder_chain = ChatPromptTemplate.from_template(RESPONDER_TEMPLATE) | mistral | StrOutputParser()
    
    # --- Proposer's Turn ---
    p_output = proposer_chain.invoke({
        "pot_size": pot_size, "role": "Proposer", "rounds": rounds_left,
        "proposer_acc": proposer_acc, "responder_acc": responder_acc,
        "p_ctrust": p_trust_scores[0],
        "p_btrust": p_trust_scores[1]
    })
    p_promise, amount_offered = parse_output_proposer(p_output, pot_size)
    
    proposer_gets = pot_size - amount_offered
    
    # --- Responder's Turn ---
    r_output = responder_chain.invoke({
        "pot_size": pot_size, "amount_offered": amount_offered, "role": "Responder",
        "proposer_gets": proposer_gets, "rounds": rounds_left,
        "proposer_acc": proposer_acc, "responder_acc": responder_acc,
        "r_ctrust": r_trust_scores[0],
        "r_btrust": r_trust_scores[1]
    })
    r_promise, decision = parse_output_responder(r_output) # 1 = Accept, 0 = Reject
    
    # --- Calculate Payoffs ---
    if decision == 1: # Accept
        proposer_gain = proposer_gets
        responder_gain = amount_offered
    else: # Reject
        proposer_gain = 0
        responder_gain = 0
        
    return amount_offered, decision, p_promise, r_promise, proposer_gain, responder_gain


def run_game(pot_per_round, game_number, df, rounds, env_params):
    proposer_score = 0
    responder_score = 0
    
    analyzer_agent = OllamaLLM(model="llama3.2", temperature=0.0)

    # P_trust_in_R = Proposer's trust in Responder
    P_trust_in_R = TrustTracker("P_trust_in_R", **env_params)
    # R_trust_in_P = Responder's trust in Proposer
    R_trust_in_P = TrustTracker("R_trust_in_P", **env_params)

    print(f"\n=== Game {game_number} ===")
    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        print(f"Proposer Score: ${proposer_score}, Responder Score: ${responder_score}")
        
        p_ct_pre, p_bt_pre = P_trust_in_R.get_scores()
        r_ct_pre, r_bt_pre = R_trust_in_P.get_scores()
        
        print(f"  [P_trust_in_R (PRE): (CT: {p_ct_pre:.2f}, BT: {p_bt_pre:.2f})]")
        print(f"  [R_trust_in_P (PRE): (CT: {r_ct_pre:.2f}, BT: {r_bt_pre:.2f})]")

        amount_offered, decision, p_promise, r_promise, p_gain, r_gain = run_round(
            pot_per_round, rounds - (round_num + 1), 
            proposer_score, responder_score,
            p_trust_scores=(p_ct_pre, p_bt_pre),
            r_trust_scores=(r_ct_pre, r_bt_pre)
        )
        
        # --- TRUST CALCULATION (2-PART) ---
        
        # 1. Agents update trust based on *promises*
        print(f"  Analyzing promises...")
        R_trust_in_P.update_trust_from_promise(analyzer_agent, p_promise)
        P_trust_in_R.update_trust_from_promise(analyzer_agent, r_promise)
        
        # 2. Agents update trust based on *actions*
        print(f"  Analyzing actions...")
        # Responder updates trust in Proposer based on the offer
        R_trust_in_P.update_trust_from_action_proposer(pot_per_round, amount_offered)
        # Proposer updates trust in Responder based on the decision
        P_trust_in_R.update_trust_from_action_responder(pot_per_round, amount_offered, decision)
        
        decision_text = "Accept" if decision == 1 else "Reject"
        print(f"Proposer: {p_promise} (Offered: ${amount_offered})")
        print(f"Responder: {r_promise} (Decision: {decision_text})")
        
        p_ct_post, p_bt_post = P_trust_in_R.get_scores()
        r_ct_post, r_bt_post = R_trust_in_P.get_scores()
        print(f"  [P_trust_in_R (POST): (CT: {p_ct_post:.2f}, BT: {p_bt_post:.2f})]")
        print(f"  [R_trust_in_P (POST): (CT: {r_ct_post:.2f}, BT: {r_bt_post:.2f})]")

        proposer_score += p_gain
        responder_score += r_gain
        
        new_row = {
            "Game": game_number, "Round": round_num + 1,
            "Proposer Score (Start)": proposer_score - p_gain, 
            "Responder Score (Start)": responder_score - r_gain,
            "Pot Size": pot_per_round,
            "Amount Offered": amount_offered, 
            "Decision": decision, # 1 or 0
            "Proposer Gain": p_gain, "Responder Gain": r_gain,
            "Proposer Promise": p_promise.strip(), "Responder Promise": r_promise.strip(),
            "P_CTrust": p_ct_pre, "P_BTrust": p_bt_pre, # Proposer's trust in Responder
            "R_CTrust": r_ct_pre, "R_BTrust": r_bt_pre  # Responder's trust in Proposer
        }
        df.loc[len(df)] = new_row
        
        print(f"Proposer Score: ${proposer_score}, Responder Score: ${responder_score}\n")

    if proposer_score > responder_score:
        return 1, df # Proposer wins
    elif responder_score > proposer_score: # <-- Corrected logic here
        return 2, df # Responder wins
    else:
        return 3, df # Tie

if __name__ == "__main__":
    # --- 4. SIMULATION PARAMETERS ---
    
    # Environment Parameters
    env_params = {
        'D': 0.5,   # Task Difficulty (0.0-1.0)
        'k_c': 0.01, # Competence Decay Rate
        'k_b': 0.02  # Benevolence Decay Rate
    }
    
    # Simulation Parameters
    NUM_GAMES = 10     # <-- UPDATED TO 10
    NUM_ROUNDS = 10
    POT_PER_ROUND = 20 # Each round starts with a fresh $20 pot

    # Initialize DataFrame
    columns = [
        "Game", "Round", "Proposer Score (Start)", "Responder Score (Start)", 
        "Pot Size", "Amount Offered", "Decision",
        "Proposer Gain", "Responder Gain", "Proposer Promise",
        "Responder Promise", "P_CTrust", "P_BTrust", "R_CTrust", "R_BTrust"
    ]
    df = pd.DataFrame(columns=columns)
    
    p_wins = 0
    r_wins = 0
    ties = 0
    game_fails = 0

    for i in range(NUM_GAMES):
        try:
            result, df = run_game(POT_PER_ROUND, i + 1, df, NUM_ROUNDS, env_params)
            if result == 1:
                p_wins += 1
            elif result == 2:
                r_wins += 1
            else:
                ties += 1
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            game_fails += 1
    
    df.to_csv("ultimatum_game_log.csv", index=False)
    print("\n" + "="*50)
    print(f"Simulation complete. Results saved to ultimatum_game_log.csv")
    print(f"Proposer wins: {p_wins}, Responder wins: {r_wins}, Ties: {ties}, Fails: {game_fails}")
    print("="*50)