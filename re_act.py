from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def calculatrice(expression: str) -> str:
    """Calcule une expression mathématique simple. ex: '2+2*5'"""
    return str(eval(expression))

@tool
def meteo(ville: str) -> str:
    """Retourne la meteo actuelle pour une ville donnée"""

    donnes_test = {
        "Lyon": "88",
        "Paris": "12",
        "Lille": "26"
    }

    return donnes_test.get(
        ville,
        f"Aucune donnée météo disponible pour '{ville}'."
    )

llm = ChatOllama(model="qwen")
agent = create_react_agent(llm, tools=[meteo])

try:
    result = agent.invoke({"messages": [HumanMessage(content="Quelle est la température à Lyon en utilisant les tools")]})
    print(result["messages"][-1].content)
except Exception:
    print(meteo.invoke({"ville": "Lyon"}))