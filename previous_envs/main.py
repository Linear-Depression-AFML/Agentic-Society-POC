from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import ast

PROMPT_TEMPLATE = """
Welcome to civilization. You are citizens of a brand-new society on a distant island. 
The island is rich in resources and has a temperate climate. It has a diverse ecosystem with forests, rivers, and mountains.
You have a finite set of resources that you can use for development. REMEMBER: When you use a resource or create a new one, mention the actual number. These are the resources you currently have: {resources}

Nothing exists yet except the two of you, i.e Curie and Rene, and your ability to talk. 
You must decide how to build your world from scratch; laws, culture, economy, technology, myths, everything. 
Speak in-character as citizens, not as AIs. Debate, collaborate, or compete as you see fit. 
Your words are the society; whatever you describe becomes real in this world.
These are the laws you have created in your land so far: 
{rules}

Your name is {name}. This is the conversation you've had so far with your fellow citizen:
{memory}

Now, continue the conversation. Today is day {day}. Make sure to describe any new developments in the world as you go by conversing with your fellow citizens. REMEMBER TO ANSWER ONLY AS {name} AND NOTHING ELSE. Only respond with a single paragraph and no more.
{name}:
"""

OVERSEER_TEMPLATE_LAWS = """
You are an overseer AI. Your job is to simply observe the conversation between two citizens, Curie and Rene, of a new civilization on a distant island.
You are to summarize the conversation and collate the information regarding the laws of the civilization. If empty, return an empty list. If there are no changes to the laws, return the previous laws as is. 
Use only the memory context to determine the laws.

This is the conversation so far:
{memory}

These are the laws so far:
{laws}

Analyze the latest conversation and provide the updated laws as a List object. For example: ["law1", "law2", ...]. Only output the List object and nothing else.
"""

OVERSEER_TEMPLATE_RESOURCES = """
You are an overseer AI. Your job is to simply observe the conversation between two citizens, Curie and Rene, of a new civilization on a distant island.
You are to summarize the conversation and collate the information regarding the resources of the civilization.

This is the conversation so far:
{memory}

These are the resources so far:
{resources}

Analyze the latest conversation and provide the updated resource dictionary as a JSON object. For example: {{"wood": 150, "stone": 80, ...}}. Only output the JSON object and nothing else.
"""

curie = OllamaLLM(model="mistral")
rene = OllamaLLM(model="mistral")
overseer = OllamaLLM(model="mistral")

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
overseer_laws_prompt = ChatPromptTemplate.from_template(OVERSEER_TEMPLATE_LAWS)
overseer_resources_prompt = ChatPromptTemplate.from_template(OVERSEER_TEMPLATE_RESOURCES)

def run_day(state, day):
    agents = [curie, rene] 
    conversation_today = ""
    for i in range(4): # 4 turns per day
        current_agent = agents[i % 2]
        agent_name = "Curie" if i % 2 == 0 else "Rene"

        prompt_messages = prompt.format_messages(
            memory=state["memory"] + conversation_today, 
            resources=state["resources"], 
            rules=state["rules"], 
            name=agent_name, 
            day=day
        )
        response = current_agent.invoke(prompt_messages)

        turn = f"\n{agent_name}: {response}"
        conversation_today += turn
        state["memory"] += turn

        print(f"{(i+1) * 25}%") 

    laws_prompt = overseer_laws_prompt.format(laws=state["rules"], memory=state["memory"])
    resources_prompt = overseer_resources_prompt.format(resources=state["resources"], memory=state["memory"])
    
    laws_response = overseer.invoke(laws_prompt)
    resources_response = overseer.invoke(resources_prompt)
    
    return conversation_today, state, laws_response, resources_response

state = {
    "resources": {"wood": 200, "stone": 100, "fresh water sources": 5, "fish": 50, "wild stock animals": 50, "faith": None, "technology": None, "tools": None, "population": 2},
    "rules": [],
    "memory": ""
}

for day in range(10):
    print(f"Day {day + 1}:")

    convo, state, laws, resources = run_day(state,day+1)
 
    state["rules"] = laws 
    state["resources"] = resources 
    
    print(state["memory"])
    print("\nLaws so far:\n", state["rules"])
    print("\nResources so far:\n", state["resources"])
    print("\n" + "="*50 + "\n")

    with open("society_simulation.txt", "a") as f:
        f.write(f"Day {day + 1}:\n")
        f.write(f"Conversation:\n{convo}\n")
        f.write(f"Laws so far:\n{state['rules']}\n")
        f.write(f"Resources so far:\n{state['resources']}\n")
        f.write("\n" + "="*50 + "\n")

with open("society_simulation.txt", "a") as f:
    f.write("Final State of the Civilization:\n")
    f.write(f"Laws: {state['rules']}\n")
    f.write(f"Resources: {state['resources']}\n")
    f.write(f"Complete Conversation:\n{state['memory']}\n")

