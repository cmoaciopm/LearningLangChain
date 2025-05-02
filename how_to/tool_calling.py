from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

def multiply(a: int, b:int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
llm_with_tools = llm.bind_tools([add, multiply])

query = "What is 3 * 12?"
print(llm_with_tools.invoke(query))

query = "What is 3 * 12? Also, what is 11 + 49?"
print(llm_with_tools.invoke(query))