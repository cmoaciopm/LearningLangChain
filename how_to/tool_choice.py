import os, getpass
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

tools = [add, multiply]
"""
# Ollama 0.6.7 doesn't support "tool_choice" attribute
llm = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
"""
# qwen respects "tool_choice" attribute
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

llm_forced_to_multiply = llm.bind_tools(tools, tool_choice="multiply")
result = llm_forced_to_multiply.invoke("what is 2 + 4")
print(result)

llm_forced_to_use_tool = llm.bind_tools(tools, tool_choice="any")
result = llm_forced_to_use_tool.invoke("What day is today?")
print(result)