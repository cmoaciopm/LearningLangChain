import getpass
import os

from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

configurable_model = init_chat_model(temperature=0, configurable_fields="any")
result = configurable_model.invoke(
    "what's your name",
    config={
        "configurable": {
            "model": "qwen-plus",
            "model_provider": "openai",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.environ["QWEN_API_KEY"]
        }
    }
)
print(result)

result = configurable_model.invoke(
    "what's your name",
    config={
        "configurable": {
            "model": "qwen3:8b",
            "model_provider": "openai",
            "base_url": "http://localhost:11434/v1",
            "api_key": "123456"
        }
    }
)
print(result)

