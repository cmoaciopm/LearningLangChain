import os, getpass

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback
from langgraph.prebuilt import create_react_agent

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

"""
llm = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
"""
llm = init_chat_model(
    model="qwen-plus",
    # Deepseek v3 doesn't support function calling until 2025-05-07
    # model="deepseek-v3",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

# Create a tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return "It's sunny."

callback = UsageMetadataCallbackHandler()

tools = [get_weather]
agent = create_react_agent(llm, tools)
for step in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in Boston?"}]},
    stream_mode="values",
    config={"callbacks": [callback]}
):
    step["messages"][-1].pretty_print()

print(f"\nTotal usage: {callback.usage_metadata}")