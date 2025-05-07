import getpass
import os

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

class GetPopulation(BaseModel):
    """Get the current population in a given location"""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

llm = init_chat_model(
    temperature=0,
    model_provider="openai",
    configurable_fields=("model", "api_key", "base_url")
)
llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])

result = llm_with_tools.invoke(
"what's bigger in 2024 LA or NYC",
    config={
        "configurable": {
            "model": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.environ["QWEN_API_KEY"]
        }
    }
)
print(result)
print(result.tool_calls)

result = llm_with_tools.invoke(
    "what's bigger in 2024 LA or NYC",
    config={
        "configurable": {
            "model": "qwen3:8b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "123456"
        }
    }
)
print(result)
print(result.tool_calls)

