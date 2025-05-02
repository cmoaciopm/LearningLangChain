from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
print(ai_msg.tool_calls)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call['name'].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)
print(messages)

result = llm_with_tools.invoke(messages)
print(result)