import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END # this import is needed to define the state graph for the agent

from langgraph.prebuilt import ToolNode

#stategraph means the state of the agent, it is a dictionary that contains the question and answer


#class State(TypedDict):
 #   question: str
 #   answer: str

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# load environment and create Gemini model

load_dotenv()

#llm = ChatGoogleGenerativeAI (
 #   model = "gemini-2.5-flash",
 #   temperature = 0
#)
#print("API BASE:", os.getenv("LEARNER_API_BASE_URL"))
#print("API KEY EXISTS:", bool(os.getenv("LEARNER_API_KEY")))

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url=os.getenv("LEARNER_API_BASE_URL"),
    api_key=os.getenv("LEARNER_API_KEY")
)

#  Difining Tools
# -----------------------------
#   Calculator Tool
# -----------------------------

@tool
def calculator_tool(expression: str) -> str:
    """
    Calculate mathematical expressions
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

#------------------------------
# Weather Tool
#------------------------------

#@tool
#def weather_tool(location: str) -> str:
  #  """
   # Get the current weather information for a given location.
   # """
    #return f"The weather in {location} is sunny and 32°C."
    #return f"Weather API will be called for: {location}"
@tool
def weather_tool(location: str) -> str:
    """
    Get the current weather information for a given location.
    """

    import requests

    # 1. Find the latitude and longitude of the city
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_response = requests.get(
        geo_url,
        params={
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }
    )

    geo_data = geo_response.json()

    if "results" not in geo_data:
        return f"Could not find the location: {location}"

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    # 2. Get current weather using the coordinates
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_response = requests.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }
    )

    weather_data = weather_response.json()

    # 3. Return the weather information
    temperature = weather_data["current"]["temperature_2m"]
    weather_code = weather_data["current"]["weather_code"]

    return (
        f"Current weather in {location}: "
        f"{temperature}°C, weather code {weather_code}."
    )
# ------------------------------
# Tools
# ------------------------------

tools = [calculator_tool, weather_tool] 

tool_node = ToolNode(tools)

# --------------------------
# Bind the tool to the LLM
# --------------------------

llm_with_tools = llm.bind_tools(tools)

graph = StateGraph(State) # this will create a state graph for the agent, the state graph is a directed graph that defines the flow of the agent

# --------------------------
# Create the Agent Node
# --------------------------

 # agent node is a function that takes the state of the agent and returns the answer to the question
def agent_node(state: State):
    system_message = SystemMessage(
    content="""
    You are a helpful general-purpose assistant.

    Answer general knowledge questions directly.

    If the user question requires a mathematical calculation,
    use the calculator_tool.

    If the user asks about weather in a specific city,
    use the weather_tool.

    Do not use tools for general knowledge questions.
    """
)

    messages = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages)

    #print("TOOL CALLS:", response.tool_calls)

    return {
        "messages": [response]
    }

# Register nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
#graph.add_edge("agent", "tools")

def should_continue(state: State): #in this function we will check if the last message has a tool call, if it does we will return "tools" else we will return END
    last_message = state["messages"][-1] # -1 means

    if last_message.tool_calls:
        return "tools"

    return END


graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

if __name__ == "__main__":
    question = input("Ask your question: ")

    initial_state = {
        "messages": [
            HumanMessage(content=question)
        ]
    }

    result = app.invoke(initial_state)

    print("\nFinal Answer:")
    print(result["messages"][-1].content)