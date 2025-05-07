import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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

llm = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
"""
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)
"""
llm_with_tools = llm.bind_tools(tools)
result = llm_with_tools.invoke(
    "Whats 119 times 8 minus 20. Don't do any math yourself, only use tools for math. Respect order of operations"
)
print(result.tool_calls)
print(result)

print("=========")

examples = [
    HumanMessage("What's the product of 317253 and 128472 plus four", name="example_user"),
    AIMessage(
        "", name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ]
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "", name="example_assistant",
        tool_calls=[
            {"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}
        ]
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant"
    )
]

system = """You are bad at math but are an expert at using a calculator. 
Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}")
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
result = chain.invoke("Whats 119 times 8 minus 20")
print(result)
print(result.tool_calls)

