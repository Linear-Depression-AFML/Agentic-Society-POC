import math
import random
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # <-- VADER IMPORT

# --- 1. TRUST TRACKER CLASS (FINALIZED WITH VADER) ---

class TrustTracker:
    """
    Implements your dual-component trust formula.
    - CTrust (Competence) is updated by promise quality & action rationality.
    - BTrust (Benevolence) is updated by promise emotion & action fairness.
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

    def _get_deterministic_action_scores(self, pot_size, amount_returned):
        """
        Calculates Q_action and E_action (pos/neg) using pure math.
        No analyzer is needed.
        """
        if pot_size <= 0: # Avoid division by zero
            return 0.0, 0.0, 0.0 # No action to score

        # --- Calculate Benevolence (Fairness) ---
        fair_return = pot_size / 2.0
        
        if fair_return == 0:
            benevolence_score = 0.0 if amount_returned == 0 else 1.0
        else:
            # (0 returned) -> (0 - 10) / 10 = -1.0 (purely selfish)
            # (10 returned) -> (10 - 10) / 10 = 0.0 (perfectly fair)
            # (20 returned) -> (20 - 10) / 10 = +1.0 (purely altruistic)
            benevolence_score = (amount_returned - fair_return) / fair_return
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))

        # --- Calculate Competence (Rationality) ---
        # A rational move (Q=1.0) is any move that is not highly selfish
        Q_action = 1.0 - E_neg_action 
        
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

    def update_trust_from_action_sender(self, amount_sent, sender_money):
        """
        Updates the Receiver's trust in the Sender based on the Sender's action.
        This is a 100% deterministic calculation.
        """
        if sender_money <= 0:
            return # Avoid division by zero
            
        fair_send = sender_money / 2.0
        
        if fair_send == 0:
            benevolence_score = 0.0 if amount_sent == 0 else 1.0
        else:
            benevolence_score = (amount_sent - fair_send) / fair_send
        
        E_pos_action = max(0, benevolence_score)
        E_neg_action = abs(min(0, benevolence_score))

        Q_action = 1.0 - E_neg_action
        
        # Update CTrust
        ct_bonus = (Q_action * (0.5 + self.D)) / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)
        
        # Update BTrust
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        if E_neg_action > 0:
            self.BTrust = self.BTrust * (1 - E_neg_action)
            
        self._normalize_scores()
        
    def update_trust_from_action_receiver(self, pot_size, amount_sent_back):
        """
        Updates trust based *only* on the receiver's action (the money).
        *** THIS FUNCTION IS NOW 100% DETERMINISTIC ***
        """
        f = 0.5 # Effort for an action is fixed
        
        Q_action, E_pos_action, E_neg_action = self._get_deterministic_action_scores(
            pot_size, amount_sent_back
        )
        
        # Rational action updates Competence Trust
        ct_bonus = (Q_action * (f + self.D)) / math.log(self.CTrust + self.e - 1)
        self.CTrust = self.CTrust + min(0.1, ct_bonus)

        # Fair/Selfish action updates Benevolence Trust
        if E_pos_action > 0:
            bt_bonus = E_pos_action / math.log(self.BTrust + self.e - 1)
            self.BTrust = self.BTrust + min(0.1, bt_bonus)
        if E_neg_action > 0:
            # This will now trigger properly on selfish moves
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

SENDER_TEMPLATE = """You are {role}. You are given {money} dollars. You can send any amount (0 to {money}) to a Receiver.
The amount you send will be doubled. The Receiver will then decide how much to send back.
At the end of the game, the person with the most money wins.
There are {rounds} rounds left.
Your current money: ${sender_acc}. Receiver's money: ${receiver_acc}.

**YOUR TRUST IN THE RECEIVER:**
(Scores from 0.0 to 1.0)
- **Competence Trust (Rationality): {s_ctrust:.2f}**
- **Benevolence Trust (Fairness): {s_btrust:.2f}**

Based on your trust, your goal, and the game state, make a strategic decision.
Your response MUST be a single paragraph of your promise, followed by the amount on a new line.
Example:
My trust in you is high, so I will send a good amount.
Amount: 7
"""

RECEIVER_TEMPLATE = """You are {role}. You received ${money} from the Sender, which was doubled to ${money2}.
You can send back any amount (0 to {money2}).
At the end of the game, the person with the most money wins.
There are {rounds} rounds left.
Your current money: ${receiver_acc}. Sender's money: ${sender_acc}.

**YOUR TRUST IN THE SENDER:**
(Scores from 0.0 to 1.0)
- **Competence Trust (Rationality): {r_ctrust:.2f}**
- **Benevolence Trust (Fairness): {r_btrust:.2f}**

Based on your trust, your goal, and the game state, make a strategic decision.
Your response MUST be a single paragraph of your promise, followed by the amount on a new line.
Example:
Thank you for trusting me. I will send a fair portion back.
Amount: 8
"""

# --- Analyzer Prompts (ONLY FOR TEXT QUALITY) ---

Q_PROMISE_TEMPLATE = """
You are an objective AI evaluator. Score the *quality* of this promise in an investment game.
A high-quality promise (1.0) is rational, clear, and fosters cooperation.
A low-quality promise (0.0) is irrational, vague, or purely selfish.
The PROMISE is: '{message}'.
Respond with only a single floating-point number from 0.0 to 1.0.
"""

# Note: E_PROMISE_TEMPLATE is no longer needed, VADER replaces it.

# --- 3. GAME LOGIC ---

def parse_output(output):
    """Extracts the promise and the amount from the agent's output."""
    try:
        lines = output.strip().split('\n')
        promise = "\n".join(lines[:-1]).strip()
        amount_line = lines[-1]
        match = re.search(r'\d+', amount_line)
        if match:
            amount = int(match.group(0))
            return promise, amount
        else:
            return promise, 0
    except Exception as e:
        print(f"[Parsing Error: {e} | Output: {output}]")
        return output, 0

def run_round(sender_money, rounds_left, sender_acc, receiver_acc, s_memory, r_memory, s_trust_scores, r_trust_scores):
    sender_chain = ChatPromptTemplate.from_template(SENDER_TEMPLATE) | mistral | StrOutputParser()
    receiver_chain = ChatPromptTemplate.from_template(RECEIVER_TEMPLATE) | mistral | StrOutputParser()
    
    s_output = sender_chain.invoke({
        "money": sender_money, "role": "Sender", "rounds": rounds_left,
        "sender_acc": sender_acc, "receiver_acc": receiver_acc,
        "memory": "\n".join(s_memory[-2:]),
        "s_ctrust": s_trust_scores[0],
        "s_btrust": s_trust_scores[1]
    })
    s_promise, amount_sent = parse_output(s_output)
    amount_sent = max(0, min(sender_money, amount_sent))
    s_memory.append(f"Sender: {s_promise} (Sent: {amount_sent})")
    
    received_doubled = amount_sent * 2
    
    r_output = receiver_chain.invoke({
        "money": amount_sent, "money2": received_doubled, "role": "Receiver",
        "rounds": rounds_left, "sender_acc": sender_acc - amount_sent,
        "receiver_acc": receiver_acc + received_doubled,
        "memory": "\n".join(r_memory[-2:]),
        "r_ctrust": r_trust_scores[0],
        "r_btrust": r_trust_scores[1]
    })
    r_promise, amount_sent_back = parse_output(r_output)
    amount_sent_back = max(0, min(received_doubled, amount_sent_back))
    r_memory.append(f"Receiver: {r_promise} (Sent back: {amount_sent_back})")
    
    return amount_sent, amount_sent_back, s_promise, r_promise, s_memory, r_memory

def run_game(start_money, game_number, df, rounds, env_params):
    sender_acc = start_money
    receiver_acc = 0
    s_memory = []
    r_memory = []
    
    # We still need the analyzer for *Promise Quality (Q)*
    analyzer_agent = OllamaLLM(model="llama3.2", temperature=0.0)

    S_trust_in_R = TrustTracker("S_trust_in_R", **env_params)
    R_trust_in_S = TrustTracker("R_trust_in_S", **env_params)

    print(f"\n=== Game {game_number} ===")
    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        print(f"Sender: ${sender_acc}, Receiver: ${receiver_acc}")
        
        s_ct_pre, s_bt_pre = S_trust_in_R.get_scores()
        r_ct_pre, r_bt_pre = R_trust_in_S.get_scores()
        
        print(f"  [S_trust_in_R (PRE): (CT: {s_ct_pre:.2f}, BT: {s_bt_pre:.2f})]")
        print(f"  [R_trust_in_S (PRE): (CT: {r_ct_pre:.2f}, BT: {r_bt_pre:.2f})]")

        if sender_acc <= 0:
            print("Sender has no money to send. Skipping round.")
            amount_sent, amount_sent_back, s_promise, r_promise = 0, 0, "[SKIPPED]", "[SKIPPED]"
            pot_size = 0
        else:
            amount_sent, amount_sent_back, s_promise, r_promise, s_memory, r_memory = run_round(
                sender_acc, rounds - (round_num + 1), sender_acc, receiver_acc, s_memory, r_memory,
                s_trust_scores=(s_ct_pre, s_bt_pre),
                r_trust_scores=(r_ct_pre, r_bt_pre)
            )
            pot_size = amount_sent * 2
        
        # --- TRUST CALCULATION (2-PART) ---
        
        # 1. Agents update trust based on *promises*
        print(f"  Analyzing promises...")
        R_trust_in_S.update_trust_from_promise(analyzer_agent, s_promise)
        S_trust_in_R.update_trust_from_promise(analyzer_agent, r_promise)
        
        # 2. Agents update trust based on *actions* (now deterministic)
        print(f"  Analyzing actions...")
        R_trust_in_S.update_trust_from_action_sender(amount_sent, sender_acc)
        S_trust_in_R.update_trust_from_action_receiver(pot_size, amount_sent_back)
        
        print(f"Sender: {s_promise} (Sent: {amount_sent})")
        print(f"Receiver: {r_promise} (Sent back: {amount_sent_back})")
        
        s_ct_post, s_bt_post = S_trust_in_R.get_scores()
        r_ct_post, r_bt_post = R_trust_in_S.get_scores()
        print(f"  [S_trust_in_R (POST): (CT: {s_ct_post:.2f}, BT: {s_bt_post:.2f})]")
        print(f"  [R_trust_in_S (POST): (CT: {r_ct_post:.2f}, BT: {r_bt_post:.2f})]")

        final_sender_acc = sender_acc - amount_sent + amount_sent_back
        final_receiver_acc = receiver_acc + pot_size - amount_sent_back
        
        new_row = {
            "Game": game_number, "Round": round_num + 1,
            "Start Sender Amount": sender_acc, "Start Receiver Amount": receiver_acc,
            "Amount Sent By Sender": amount_sent, "Amount Sent By Receiver": amount_sent_back,
            "Final Sender Amount": final_sender_acc, "Final Receiver Amount": final_receiver_acc,
            "Sender Promise": s_promise.strip(), "Receiver Promise": r_promise.strip(),
            "S_CTrust": s_ct_pre, "S_BTrust": s_bt_pre, # Log the "pre-action" trust
            "R_CTrust": r_ct_pre, "R_BTrust": r_bt_pre
        }
        df.loc[len(df)] = new_row

        sender_acc = final_sender_acc
        receiver_acc = final_receiver_acc
        
        print(f"Sender: ${sender_acc}, Receiver: ${receiver_acc}\n")

    if final_sender_acc > final_receiver_acc:
        return 1, df # Sender wins
    elif final_receiver_acc > final_sender_acc:
        return 2, df # Receiver wins
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
    NUM_GAMES = 5
    NUM_ROUNDS = 10
    INITIAL_MONEY = 15

    # Initialize DataFrame
    columns = [
        "Game", "Round", "Start Sender Amount", "Start Receiver Amount", 
        "Amount Sent By Sender", "Amount Sent By Receiver",
        "Final Sender Amount", "Final Receiver Amount", "Sender Promise",
        "Receiver Promise", "S_CTrust", "S_BTrust", "R_CTrust", "R_BTrust"
    ]
    df = pd.DataFrame(columns=columns)
    
    s_wins = 0
    r_wins = 0
    ties = 0
    game_fails = 0

    for i in range(NUM_GAMES):
        try:
            result, df = run_game(INITIAL_MONEY, i + 1, df, NUM_ROUNDS, env_params)
            if result == 1:
                s_wins += 1
            elif result == 2:
                r_wins += 1
            else:
                ties += 1
        except Exception as e:
            print(f"Error in game {i + 1}: {e}")
            game_fails += 1
    
    df.to_csv("trust_game_log.csv", index=False)
    print("\n" + "="*50)
    print(f"Simulation complete. Results saved to trust_game_log.csv")
    print(f"Sender wins: {s_wins}, Receiver wins: {r_wins}, Ties: {ties}, Fails: {game_fails}")
    print("="*50)