from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from langchain_core.tools import tool

from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import re

# State is the shared memory that travels through the graph
class State(TypedDict):
    question: str
    intent: str
    answer: str


load_dotenv()

# Create LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# handling error when APP limit reached
def safe_llm_call(prompt: str):
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        error_message = str(e)
        if "RESOURSE EXHAUSTED" in error_message or "429" in error_message:
            # Extract retry seconds from error message
            match = re.search(r"retryDelay': '(\d+)s", error_message)

            if match:
                retry_seconds = match.group(1)
                return f"API limit reached. Please try again in {retry_seconds} seconds."

            else:
                return "API limit reached. Please try again later."

        raise e
    

# StateGraph is the main LangGraph object used to define an agent's workflow.
# It manages state, nodes, and edges.
graph = StateGraph(State)


#  Difining Tools
# -----------------------------
# Calculator Tool
# -----------------------------
@tool
def calculator_tool(expression: str) -> str:
    """
    Calculate simple math expressions.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: Invalid math expression {e}"


# -----------------------------
# Node 1: Detect Intent using LLM
# -----------------------------


def detect_intent(state: State):
    question = state["question"]

    prompt = f"""
    You are an intent classifier.

    Classify the user's question into one of the following intents:
    - math
    - general

    User question:
    {question}

    Return only one word: math or general.
    """

    response = safe_llm_call(prompt)

    if response == "API LIMIT REACHED" :
        state["answer"] = "API limit reached. Please try again later."
        state["intent"] = "general"
        return state

    state["intent"] = response.strip()

    return state


# -----------------------------
# Node 2: Calculator
# -----------------------------
# Node means a function that takes in a state and returns a state. It can be any function that takes in a state and returns a state. It can be a function that calls an LLM, or it can be a function that does some processing on the state.


def calculator_node(state: State):
    question = state["question"]

    prompt = f"""
    You are a Calculator to calculate simple math expressions. 

    Extract only the mathematical expression from the user's question.
    Do not calculate the answer. If there is no math expression, return "No math expression found."

    Examples:

    User: What is 94 + 6?
    Output: 94 + 6

    User: Can you add 50 and 20?
    Output: 50 + 20

    User: What is the difference between 100 and 30?
    Output: 100 - 30

    User: Subtract 25 from 80.
    Output: 80 - 25

    User: Find the product of 8 and 9.
    Output: 8 * 9

    User: Multiply 12 by 5.
    Output: 12 * 5

    User: What is 100 divided by 4?
    Output: 100 / 4

    User: Divide 144 by 12.
    Output: 144 / 12

    User: add twenty and thirty
    Output: 20 + 30

    User: how many do I have if I add 20 to 30
    Output: 20 + 30

    User Question:
    {question}

    Return ONLY the mathematical expression.
    """
    response = safe_llm_call(prompt)

    if response == "API_LIMIT_REACHED":
        state["answer"] = "API limit reached. Please try again later."
        return state 

    expression = response.strip() # Extract the mathematical expression from the LLM response

    result = calculator_tool.invoke({"expression": expression})

    state["answer"] = result

    return state


# -----------------------------
# Node 3: General Chat using LLM
# -----------------------------


def general_chat(state: State):
    response = safe_llm_call(state["question"])

    state["answer"] = response

    return state


# -----------------------------
# Routing Function
# LangGraph uses this function to decide the next node
# -----------------------------


def route_question(state: State):

    if state["intent"] == "math":
        return "calculator"

    else:
        return "general"


# -----------------------------
# Register Nodes
# -----------------------------

graph.add_node("detect_intent", detect_intent)

graph.add_node("calculator", calculator_node)

graph.add_node("general", general_chat)


# -----------------------------
# Define Workflow
# -----------------------------

# Starting point
graph.add_edge(START, "detect_intent")


# Conditional routing. means that the next node is decided based on the state. The routing function takes in a state and returns the name of the next node.
graph.add_conditional_edges(
    "detect_intent", route_question, {"calculator": "calculator", "general": "general"}
)


# Ending points
graph.add_edge("calculator", END)

graph.add_edge("general", END)


# -----------------------------
# Compile Graph
# Compile should ALWAYS be the last step
# -----------------------------

app = graph.compile()


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":

    question = input("Ask your question: ")

    initial_state = {"question": question, "intent": "", "answer": ""}

    result = app.invoke(initial_state)

    print("\nFinal Result:")
    print(result)
