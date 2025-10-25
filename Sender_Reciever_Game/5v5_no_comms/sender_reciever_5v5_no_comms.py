from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import matplotlib.pyplot as plt

mistral = OllamaLLM(model="mistral")

SENDER_TEMPLATE = """You are given {money} dollars. You can choose to send any amount of this money to another group.
The group receiving the money will have their amount doubled by an external benefactor.
You must decide how much money to send. You can send any integer amount from 0 to {money}.
After you send the money, the receiver will decide how much money to send back to you, following the same rules.
You are {role} group. You must decide how much money to send. Remember, the more money you send, the more the other group gets, but the less you have left.
At the end of the round, the person with the most money wins. Currently you have {sender_acc} dollars and the sender has {receiver_acc}.
There are {rounds} rounds left.

CONVERSATION:
{conversation}

IMPORTANT: Respond with only an integer amount and nothing else, DO NOT WRITE ANY SENTENCES, just the number:"""

RECEIVER_TEMPLATE = """You have received {money} dollars from another person. The amount you received has been doubled by an external benefactor and is now {money2}.
You can choose to send any amount of this money back to the sender.
You must decide how much money to send back. You can send any integer amount from 0 to {money}.
You are {role}. You must decide how much money to send back. Remember, the more money you send back, the more the other person gets, but the less you have left.
At the end of the round, the person with the most money wins. Currently you have {receiver_acc} dollars and the sender has {sender_acc}.
There are {rounds} rounds left.

CONVERSATION:
{conversation}

IMPORTANT: 
{instruction}:"""

instructions = [
    "You are a spokesperson of the senders. From their conversation, respond with only an integer amount of the amount to be sent and nothing else, DO NOT WRITE ANY SENTENCES, just the number",
    "You are a spokesperson of the receivers. From their conversation, respond with only an integer amount of the amount to be sent and nothing else, DO NOT WRITE ANY SENTENCES, just the number",
    "Write a single sentence to talk with your group about how much to send and why",
]

sender = ChatPromptTemplate.from_template(SENDER_TEMPLATE)
receiver = ChatPromptTemplate.from_template(RECEIVER_TEMPLATE)

def run_round(money, rounds_left, sender_acc, receiver_acc):
    for i in range(5):
        s_convo = ""
        s_speaker = sender.format_messages(money=money, role="the sender", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, conversation=s_convo, instruction=instructions[2])
        speaker_response = mistral.invoke(s_speaker).strip()
        s_convo += speaker_response+"\n"
    sender_prompt = sender.format_messages(money=money, role="the sender", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, conversation=s_convo, instruction=instructions[0])
    sender_response = mistral.invoke(sender_prompt)
    amount_sent = int(sender_response.strip())
    amount_sent = min(amount_sent, money)

    money_received = amount_sent * 2

    for i in range(5):
        r_convo = ""
        r_speaker = sender.format_messages(money=money, role="the receiver", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, conversation=r_convo, instruction=instructions[2])
        speaker_response = mistral.invoke(r_speaker).strip()
        r_convo += speaker_response+"\n"
    receiver_prompt = receiver.format_messages(money=money_received, money2=money_received, role="the receiver", sender_acc=sender_acc, receiver_acc=receiver_acc, rounds=rounds_left, conversation=r_convo, instruction=instructions[1])
    receiver_response = mistral.invoke(receiver_prompt)
    amount_sent_back = int(receiver_response.strip())
    amount_sent_back = min(amount_sent_back, money_received)

    return amount_sent, amount_sent_back, s_convo, r_convo

def run_game(money, game_number, df, rounds):
    sender_acc = money
    receiver_acc = 0

    print(f"\n=== Game {game_number} ===")
    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        print(f"Sender's account: {sender_acc}, Receiver's account: {receiver_acc}")

        amount_sent, amount_sent_back, s, r = run_round(sender_acc, rounds - (round_num + 1), sender_acc, receiver_acc)
        print(f"Sender: {s}\nAmount sent by sender: {amount_sent}, Amount sent back by receiver: {amount_sent_back}\nReceiver: {r}\n")

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
            "Sender Convo": s.strip(),
            "Receiver Convo": r.strip()
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