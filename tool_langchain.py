from typing import TypedDict, Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

llm = ChatOllama(model="qwen:latest", temperature=0)

class State(TypedDict):
    question: str
    categorie: str
    response: str


def router(state: State) -> State:
    prompt = (
        "Classifie la question en 'calcul' ou 'recherche'. "
        f"Question: {state['question']}"
    )

    category = llm.invoke(prompt).content.strip().lower()

    return {**state, "categorie": category}


def agent_calcul(state: State) -> State:
    prompt = f"Résous ce calcul et donne uniquement le résultat : {state['question']}"
    response = llm.invoke(prompt)

    return {**state, "response": response.content}


def agent_recherche(state: State) -> State:
    prompt = f"Réponds de manière concise : {state['question']}"
    response = llm.invoke(prompt)

    return {**state, "response": response.content}


def choisir_agent(state: State) -> Literal["agent_calcul", "agent_recherche"]:
    cat = state["categorie"].lower()
    return "agent_calcul" if "calcul" in cat else "agent_recherche"


graph = StateGraph(State)

graph.add_node("router", router)
graph.add_node("agent_calcul", agent_calcul)
graph.add_node("agent_recherche", agent_recherche)

graph.set_entry_point("router")

graph.add_conditional_edges("router", choisir_agent)

graph.add_edge("agent_calcul", END)
graph.add_edge("agent_recherche", END)

app = graph.compile()

result = app.invoke({
    "question": "Combien font 5% de 80 ?",
    "categorie": "",
    "response": ""
})

print(result["response"])