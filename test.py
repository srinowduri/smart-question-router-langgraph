from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    question: str
    intent: str #intent means the purpose of the question, e.g., "ask for information", "request action", etc.
    answer: str

def detect_intent(state: State):
    question = state["question"].lower()
    
    if any(op in question for op in ["+", "-", "*", "/"]):
        state["intent"] = "math"
    else:
        state["intent"] = "general"

    return state

def calculator(state: State):
    question = state["question"]

    numbers = []

    for word in question.split():
        if word.isdigit():
            numbers.append(int(word))
    
    result = numbers[0] + numbers[1]  

    state["answer"] = str(result)

    return state


if __name__ == "__main__":

    state = {
        "question": input("Ask your question: "),
        "intent": "",
        "answer": ""
    }

    state = detect_intent(state)

    print("After Intent Detection:")
    print(state)

    #state = calculator(state)

    #print("After Calculator:")
    #print(state)

graph = StateGraph()

graph.add_node("detect_intent", detect_intent)
graph.add_node("calculator", calculator)

