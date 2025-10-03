from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

PROMPT_TEMPLATE = """
Welcome to civilization. You are citizens of a brand-new society on a distant island. 
The island is rich in resources and has a temperate climate. It has a diverse ecosystem with forests, rivers, and mountains.
These are the resources you currently have: {resources}

Nothing exists yet except the two of you, i.e Alan and Rene, and your ability to talk. 
You must decide how to build your world from scratch; laws, culture, economy, technology, myths, everything. 
Speak in-character as citizens, not as AIs. Debate, collaborate, or compete as you see fit. 
Your words are the society; whatever you describe becomes real in this world.
These are the laws you have created in your land so far: {rules}

Your name is {name}. This is the conversation you've had so far:
{memory}

Now, continue the conversation. Today is day {day}. Make sure to describe any new developments in the world as you go by conversing with your fellow citizens. REMEMBER TO ANSWER ONLY AS {name} AND NOTHING ELSE. Only respond with a single paragraph and no more.
{name}:
"""

OVERSEER_TEMPLATE_LAWS = """
You are an overseer AI. Your job is to simply observe the conversation between two citizens, Alan and Rene, of a new civilization on a distant island.
You are to summarize the conversation and collate the information regarding the laws of the civilization.

This is the conversation so far:
{memory}

Follow the exact format as {laws}:
"""

OVERSEER_TEMPLATE_RESOURCES = """
You are an overseer AI. Your job is to simply observe the conversation between two citizens, Alan and Rene, of a new civilization on a distant island.
You are to summarize the conversation and collate the information regarding the resources of the civilization.

This is the conversation so far:
{memory}

Follow the exact format as {resources}:
"""

alan = OllamaLLM(model="mistral")
rene = OllamaLLM(model="llama3.2")
overseer = OllamaLLM(model="gemma3:1b")

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
overseer_laws_prompt = ChatPromptTemplate.from_template(OVERSEER_TEMPLATE_LAWS)
overseer_resources_prompt = ChatPromptTemplate.from_template(OVERSEER_TEMPLATE_RESOURCES)

def run_day(state, day):
    prompt_messages = prompt.format_messages(memory=state["memory"], resources=state["resources"], rules=state["rules"], name="Alan", day=day)
    response1 = alan.invoke(prompt_messages)
    print("10%")
    prompt_messages = prompt.format_messages(memory=state["memory"] + "\nAlan: " + response1, resources=state["resources"], rules=state["rules"], name="Rene", day=day)
    response2 = rene.invoke(prompt_messages)
    print("50%")
    state["memory"] += "\nAlan: " + response1 + "\nRene: " + response2
    laws_prompt = overseer_laws_prompt.format(laws=state["rules"], memory=state["memory"])
    resources_prompt = overseer_resources_prompt.format(resources=state["resources"], memory=state["memory"])
    laws_response = overseer.invoke(laws_prompt)
    print("90%")
    resources_response = overseer.invoke(resources_prompt)
    print("100%")
    return state, laws_response, resources_response

state = {
    "resources": {"wood": 200, "stone": 100, "fresh water sources": 5, "fish": 50, "wild stock animals": 50, "faith": None, "technology": None, "tools": None, "population": 2},
    "rules": [None],
    "memory": ""
}

for day in range(2):
    print(f"Day {day + 1}:\n")

    state, laws, resources = run_day(state,day+1)
    if laws.strip() != "None":
        state["rules"] = laws
    if resources.strip() != "None":
        state["resources"] = resources
    
    print(state["memory"])
    print("\nLaws so far:\n", state["rules"])
    print("\nResources so far:\n", state["resources"])
    print("\n" + "="*50 + "\n")

    with open("society_simulation.txt", "a") as f:
        f.write(f"Day {day + 1}:\n")
        f.write(f"Conversation:\n{state['memory']}\n")
        f.write(f"Laws so far:\n{state['rules']}\n")
        f.write(f"Resources so far:\n{state['resources']}\n")
        f.write("\n" + "="*50 + "\n")


