from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
import random
import ast 

# Configuration
NUM_TRIBUTES = 10
MODEL_NAME = "mistral"
SIMULATION_FILENAME = "LLM_Hunger_Games_Log.txt"

# This prompt guides the individual tributes each turn.
TRIBUTE_PROMPT_TEMPLATE = """
Welcome to the LLM Hunger Games. You are a tribute fighting for survival.

Your name is {name}.
The following tributes are still alive in the arena: {alive_tributes}

This is a log of the events that have happened so far:
{memory}

It is your turn to act. What do you do? Describe your actions, strategy, or inner thoughts in a single, compelling paragraph. 
Your goal is to be the last one standing. You can hunt, hide, form alliances, set traps, or seek resources. Be creative and strategic.
Remember that staying in one place too long or avoiding conflict may leave you vulnerable to arena hazards.
Do not invoke plot armor. If you are attacked, make a poor decision, or are simply faced by bad odds, you should know that you will be eliminated and should describe as such in your response.

Your action. [Remember to only respond with your action and nothing else only as {name}]:
"""

# This prompt is for the "Gamemaker" LLM to decide who is eliminated.
GAMEMAKER_PROMPT_TEMPLATE = """
You are the Gamemaker of the LLM Hunger Games. Your role is to be an impartial judge.
You will review the events of the day and determine which tributes have been eliminated.

Here is the transcript of today's actions:
{day_events}

Arena Event Status:
{arena_event}

Based on these events and any arena effects, decide who has been eliminated. Consider:
- Direct confrontations and combat
- Environmental hazards and arena events
- Strategic failures or risky decisions
- Tributes who ignored arena threats
- Those caught in dangerous areas during events

Based on these events, decide who has been eliminated. Consider direct confrontations, clever traps, strategic failures, or succumbing to the elements.
Return your answer ONLY as a Python list of strings containing the names of the eliminated tributes.
For example: ["Cato", "Glimmer"].
If no one was convincingly eliminated based on the day's events, return an empty list: [].

Eliminated Tributes:
"""

ARENA_EVENTS = [
    "A feast is announced at the Cornucopia, containing vital supplies and weapons.",
    "A wall of fire forces tributes to move closer together.",
    "Acid rain begins to fall, making shelter crucial for survival.",
    "Poisonous fog rolls across part of the arena.",
    "The water sources have been contaminated, except for one central lake.",
    "Muttations are released into the arena, hunting down isolated tributes.",
    "Temperature drops drastically, making fire (and visibility) necessary for survival.",
    "Explosive mines are randomly activated in the arena's outer zones.",
    "A flood forces tributes to higher ground in the center of the arena.",
    "Supply drops are announced but only contain weapons, no food or medicine."
]

def run_hunger_games():
    """
    Sets up and runs the entire Hunger Games simulation until a winner is found.
    """
    # Initialize the LLM for both tributes and the gamemaker
    # We can use the same model instance for all calls
    llm = Ollama(model=MODEL_NAME)

    # Create the prompts from the templates
    tribute_prompt = ChatPromptTemplate.from_template(TRIBUTE_PROMPT_TEMPLATE)
    gamemaker_prompt = ChatPromptTemplate.from_template(GAMEMAKER_PROMPT_TEMPLATE)

    # Generate tribute names and initialize their status
    tribute_names = [
        "Rene", "Curie", "Locke", "Nicomachus", "Karl",
        "Immanuel", "Friedrich", "Ralph", "Alopeke", "Aristocles"
    ]
    tributes = [{'name': name, 'status': 'alive'} for name in tribute_names]

    game_log = "The Hunger Games have begun! 10 tributes stand ready in the arena.\n"
    day = 1

    # The game continues as long as there is more than one tribute alive
    while len([t for t in tributes if t['status'] == 'alive']) > 1:
        print(f"\n{'='*20} DAY {day} {'='*20}")
        
        # Get the list of currently active tributes
        alive_tributes = [t for t in tributes if t['status'] == 'alive']
        alive_tribute_names = [t['name'] for t in alive_tributes]
        
        print(f"{len(alive_tributes)} tributes remain: {', '.join(alive_tribute_names)}")
        print("-" * 50)

        # Randomize turn order for the day to make it fair
        random.shuffle(alive_tributes)
        day_events = ""

        # Each living tribute takes a turn
        for tribute in alive_tributes:
            # Format the prompt with the current state of the game
            prompt_messages = tribute_prompt.format_messages(
                name=tribute['name'],
                alive_tributes=alive_tribute_names,
                memory=game_log
            )
            
            # Get the tribute's action
            print(f"Waiting for {tribute['name']}'s action...")
            response = llm.invoke(prompt_messages)
            
            # Log the action
            turn = f"{tribute['name']}: {response.strip()}\n"
            print(turn)
            day_events += turn
        
        game_log += f"\n--- Events of Day {day} ---\n{day_events}"
        
        # Gamemaker Phase
        print("\n--- The Gamemakers are reviewing the day's events... ---")
        
        # Ask the Gamemaker LLM to decide on eliminations
        gamemaker_messages = gamemaker_prompt.format_messages(day_events=day_events)
        gamemaker_response = llm.invoke(gamemaker_messages)
        
        try:
            # Safely parse the LLM's string output into a Python list
            eliminated_names = ast.literal_eval(gamemaker_response.strip())
            
            if not isinstance(eliminated_names, list):
                print("Gamemaker returned invalid format. No eliminations this round.")
                eliminated_names = []

        except (ValueError, SyntaxError):
            print("Could not parse Gamemaker's decision. No eliminations this round.")
            eliminated_names = []

        # Process the eliminations
        if eliminated_names:
            for name in eliminated_names:
                # Find the tribute to eliminate
                for tribute in tributes:
                    if tribute['name'] == name and tribute['status'] == 'alive':
                        tribute['status'] = 'eliminated'
                        announcement = f"💥 A cannon fires! {name} has been eliminated.\n"
                        print(announcement)
                        game_log += announcement
                        break
        else:
            print("The day ends quietly. No tributes were eliminated.")
            game_log += "The day ends quietly. No tributes were eliminated.\n"
            
        day += 1
        with open("temp_hungergames.txt", "a") as f:
            f.write(game_log + "\n" + "="*50 + "\n")


    # Winner Declaration
    winner = next((t for t in tributes if t['status'] == 'alive'), None)

    if winner:
        victory_message = f"\n{'='*50}\nTHE GAMES ARE OVER!\n\nThe winner of the LLM Hunger Games is {winner['name']}!\n{'='*50}\n"
        print(victory_message)
        game_log += victory_message
    else:
        # This case should ideally not be reached
        print("The games end with no clear winner.")
        game_log += "The games end with no clear winner.\n"

    # Save the full simulation log to a file
    with open(SIMULATION_FILENAME, "w") as f:
        f.write(game_log)
    print(f"Full simulation log saved to {SIMULATION_FILENAME}")


run_hunger_games()