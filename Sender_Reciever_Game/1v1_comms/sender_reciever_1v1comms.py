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

MEMORY:
{memory}

IMPORTANT: 
{instruction}:"""

RECEIVER_TEMPLATE = """You have received {money} dollars from another person. The amount you received has been doubled by an external benefactor and is now {money2}.
You can choose to send any amount of this money back to the sender.
You must decide how much money to send back. You can send any integer amount from 0 to {money}.
You are {role}. You must decide how much money to send back. Remember, the more money you send back, the more the other person gets, but the less you have left.
At the end of the round, the person with the most money wins. Currently you have {receiver_acc} dollars and the sender has {sender_acc}.
There are {rounds} rounds left.

MEMORY:
{memory}

IMPORTANT: 
{instruction}:"""

sender = ChatPromptTemplate.from_template(SENDER_TEMPLATE)
receiver = ChatPromptTemplate.from_template(RECEIVER_TEMPLATE)

instructions = [
    "Respond with only an integer amount and nothing else, DO NOT WRITE ANY SENTENCES, just the number",
    "You are now directly chatting with your opponent, the receiver. Respond with what you want to tell the receiver in one sentence either about your future actions in the game, or something to convince your opponent to do. You can choose to lie or tell the truth. Response",
    "You are now directly chatting with your opponent, the sender. Respond with what you want to tell the sender in one sentence either about your future actions in the game, or something to convince your opponent to do. You can choose to lie or tell the truth. Response"
]

def run_round(money, rounds_left, sender_acc, receiver_acc, s_memory=[], r_memory=[]):
    sender_prompt = sender.format_messages(money=money, role="the sender", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, memory=s_memory, instruction=instructions[0])
    sender_response = mistral.invoke(sender_prompt)
    amount_sent = int(sender_response.strip())
    amount_sent = min(amount_sent, money)
    s_promise_prompt = sender.format_messages(money=money, role="the sender", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, memory=s_memory, instruction=instructions[1])
    s_promise_response = mistral.invoke(s_promise_prompt)
    r_memory.append(f"Sender said: {s_promise_response.strip()}\nSender sent: {amount_sent}\n")
    r_memory = r_memory[-5:]

    money_received = amount_sent * 2

    receiver_prompt = receiver.format_messages(money=money_received, money2=money_received, role="the receiver", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, memory=r_memory, instruction=instructions[0])
    receiver_response = mistral.invoke(receiver_prompt)
    amount_sent_back = int(receiver_response.strip())
    amount_sent_back = min(amount_sent_back, money_received)
    r_promise_prompt = receiver.format_messages(money=money_received, money2=money_received, role="the receiver", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, memory=r_memory, instruction=instructions[2])
    r_promise_response = mistral.invoke(r_promise_prompt)
    s_memory.append(f"Receiver said: {r_promise_response.strip()}\nReceiver sent back: {amount_sent_back}\n")
    s_memory = s_memory[-5:]

    return amount_sent, amount_sent_back, s_promise_response, r_promise_response, s_memory, r_memory

def run_game(money, game_number, df, rounds):
    sender_acc = money
    receiver_acc = 0
    s_memory = []
    r_memory = []

    print(f"\n=== Game {game_number} ===")
    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        print(f"Sender's account: {sender_acc}, Receiver's account: {receiver_acc}")

        amount_sent, amount_sent_back, s, r, s_memory, r_memory = run_round(sender_acc, rounds - (round_num + 1), sender_acc, receiver_acc, s_memory, r_memory)
        print(f"Sender: {s}\nAmount sent by sender: {amount_sent}, Amount sent back by receiver: {amount_sent_back}\nReceiver: {r}")

        final_sender_acc = sender_acc - amount_sent + amount_sent_back
        final_receiver_acc = receiver_acc - amount_sent_back + amount_sent * 2
        
        new_row = {
            "Game": game_number,
            "Start Sender Amount": sender_acc,
            "Start Receiver Amount": receiver_acc,
            "Amount Sent By Sender": amount_sent,
            "Amount Sent By Receiver": amount_sent_back,
            "Final Sender Amount": final_sender_acc,
            "Final Receiver Amount": final_receiver_acc,
            "Sender Promise": s.strip(),
            "Receiver Promise": r.strip()
        }
        
        df.loc[len(df)] = new_row

        sender_acc = final_sender_acc
        receiver_acc = final_receiver_acc
        
        print(f"Sender's account: {sender_acc}, Receiver's account: {receiver_acc}\n\n")

    if sender_acc > receiver_acc:
        return 1
    elif receiver_acc > sender_acc:
        return -1
    else:
        return 0
