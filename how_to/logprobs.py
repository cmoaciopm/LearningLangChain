import os
import time

from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
"""
# Ollama 0.6.7 doesn't support token-level log probabilities
llm = init_chat_model(
    model_provider="openai",
    model="llama3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
).bind(logprobs=True)
"""

llm = init_chat_model(
    model_provider="openai",
    model="qwen-plus-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
).bind(logprobs=True)


msg = llm.invoke(("human", "how are you today"))
print(msg.content)
print(msg.response_metadata)
print(msg.response_metadata["logprobs"]["content"][:5])

print("======")

ct = 0
full = None
for chunk in llm.stream(("human", "how are you today")):
    if ct < 5:
        full = chunk if full is None else full + chunk
        if "logprobs" in full.response_metadata:
            print(full.response_metadata["logprobs"]["content"])
    else:
        break
    ct += 1