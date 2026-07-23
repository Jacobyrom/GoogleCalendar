import asyncio 

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient(
        {
            "catalogue": {
                "command": "python",
                "args":["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    print("TOOLS DISPONIBLES :")
    for t in tools:
        print("-", t.name)

        llm = init_chat_model(
            model="qwen:latest",
            model_provider="ollama",
            base_url="http://localhost:11434"
        )
        
        agent = create_agent(
            model=llm,
            tools=tools,
        )

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Bonjour, donne moi le produit le moins cher"
                }
            ]
        }
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())