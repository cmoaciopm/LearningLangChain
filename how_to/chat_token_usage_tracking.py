import os, getpass

from langchain.chat_models import init_chat_model

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

response = llm.invoke("hello")
print(response)
print(response.usage_metadata)

aggregate = None
for chunk in llm.stream("hello", stream_usage=True):
    print(chunk)
    aggregate = chunk if aggregate is None else aggregate + chunk
print(aggregate.content)
print(aggregate.usage_metadata)