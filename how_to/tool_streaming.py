import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

tools = [add, multiply]
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

llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"
for chunk in llm_with_tools.stream(query):
    print(chunk.tool_call_chunks)

print("=========")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk
    print(gathered.tool_call_chunks)

print("=========")

first = True
for chunk in llm_with_tools.stream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk
    print(gathered.tool_calls)