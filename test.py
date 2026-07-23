from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(model="qwen")

history = []

def chat (user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content

print(chat("Explique moi la différence entre un chat et un chien."))

print(chat("Peux-tu me donner un exemple de phrase avec le mot 'chat' ?"))