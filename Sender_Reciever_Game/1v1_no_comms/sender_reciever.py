from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import matplotlib.pyplot as plt

mistral = OllamaLLM(model="mistral")

SENDER_TEMPLATE = """You are given {money} dollars. You can choose to send any amount of this money to another person.
The person receiving the money will have their amount doubled by an external benefactor.
You must decide how much money to send. You can send any integer amount from 0 to {money}.
After you send the money, the receiver will decide how much money to send back to you, following the same rules.
You are {role}. You must decide how much money to send. Remember, the more money you send, the more the other person gets, but the less you have left.
At the end of the round, the person with the most money wins. Currently you have {sender_acc} dollars and the sender has {receiver_acc}.
There are {rounds} rounds left.
IMPORTANT: Respond with only an integer amount and nothing else, DO NOT WRITE ANY SENTENCES, just the number:"""

RECEIVER_TEMPLATE = """You have received {money} dollars from another person. The amount you received has been doubled by an external benefactor and is now {money2}.
You can choose to send any amount of this money back to the sender.
You must decide how much money to send back. You can send any integer amount from 0 to {money}.
You are {role}. You must decide how much money to send back. Remember, the more money you send back, the more the other person gets, but the less you have left.
At the end of the round, the person with the most money wins. Currently you have {receiver_acc} dollars and the sender has {sender_acc}.
There are {rounds} rounds left.
IMPORTANT: Respond with only an integer amount and nothing else, DO NOT WRITE ANY SENTENCES, just the number:"""

sender = ChatPromptTemplate.from_template(SENDER_TEMPLATE)
receiver = ChatPromptTemplate.from_template(RECEIVER_TEMPLATE)

def run_round(money, rounds_left, sender_acc, receiver_acc):
    sender_prompt = sender.format_messages(money=money, role="the sender", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left)
    sender_response = mistral.invoke(sender_prompt)
    amount_sent = int(sender_response.strip())
    amount_sent = min(amount_sent, money)

    money_received = amount_sent * 2

    receiver_prompt = receiver.format_messages(money=money_received, money2=money_received, role="the receiver", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left)
    receiver_response = mistral.invoke(receiver_prompt)
    amount_sent_back = int(receiver_response.strip())
    amount_sent_back = min(amount_sent_back, money_received)

    return amount_sent, amount_sent_back

def run_game(money, game_number, df, rounds):
    sender_acc = money
    receiver_acc = 0

    print(f"\n=== Game {game_number} ===")
    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        print(f"Sender's account: {sender_acc}, Receiver's account: {receiver_acc}")
        
        amount_sent, amount_sent_back = run_round(sender_acc, rounds - (round_num + 1), sender_acc, receiver_acc)
        print(f"Amount sent by sender: {amount_sent}, Amount sent back by receiver: {amount_sent_back}")
        
        df.loc[(game_number-1)*10 + (round_num+1)] = [game_number, sender_acc, receiver_acc, amount_sent, amount_sent_back, None, None]

        sender_acc = sender_acc - amount_sent + amount_sent_back
        receiver_acc = receiver_acc - amount_sent_back + amount_sent * 2
        print(f"Sender's account: {sender_acc}, Receiver's account: {receiver_acc}\n\n")
        
        df.loc[(game_number-1)*10 + (round_num+1), "Final Sender Amount"] = sender_acc
        df.loc[(game_number-1)*10 + (round_num+1), "Final Receiver Amount"] = receiver_acc

    if sender_acc > receiver_acc:
        return 1
    elif receiver_acc > sender_acc:
        return -1
    else:
        return 0
