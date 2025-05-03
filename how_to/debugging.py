from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain.globals import set_verbose
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

set_verbose(True)
# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({
    "input": "Who directed the 2023 film Oppenheimer and what is their age in days?"
})
print(result)