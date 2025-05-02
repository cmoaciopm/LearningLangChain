from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticToolsParser
from pydantic import Field, BaseModel

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

class add(BaseModel):
    """Add two integers."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

class multiply(BaseModel):
    """Multiply two integers."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
llm_with_tools = llm.bind_tools([add, multiply])

query = "What is 3 * 12? Also, what is 11 + 49?"
chain = llm_with_tools | PydanticToolsParser(tools=[add, multiply])
print(chain.invoke(query))