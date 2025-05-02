from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply a and B.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

llm_with_tools = llm.bind_tools([multiply])

result = llm_with_tools.invoke("Hello world!")
print(result)

result = llm_with_tools.invoke("What is 2 multiplied by 3?")
print(result)