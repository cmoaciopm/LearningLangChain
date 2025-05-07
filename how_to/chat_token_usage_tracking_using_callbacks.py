import os, getpass

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

"""
llm = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
"""
llm = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

callback = UsageMetadataCallbackHandler()
llm.invoke("hello", config={"callbacks": [callback]})
llm.invoke("What's your name?", config={"callbacks": [callback]})
llm.invoke("Tell me a joke", config={"callbacks": [callback]})
print(callback.usage_metadata)