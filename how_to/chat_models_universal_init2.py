import getpass
import os

from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

first_llm = init_chat_model(
    model="qwen-plus",
    temperature=0,
    configurable_fields=("model", "model_provider", "temperature", "max_tokens", "api_key", "base_url"),
    config_prefix="first" # useful when you have a chain with multiple models
)
result = first_llm.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model_provider": "openai",
            "first_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "first_api_key": os.environ["QWEN_API_KEY"]
        }
    }
)
print(result)

result = first_llm.invoke(
    "what's your name",
    config={
        "configurable": {
            "first_model": "qwen3:8b",
            "first_model_provider": "openai",
            "first_base_url": "http://localhost:11434/v1",
            "first_api_key": "123456",
            "first_temperature": 0.5,
            "first_max_tokens": 100
        }
    }
)
print(result)

