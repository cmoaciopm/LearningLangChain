import time

from langchain.chat_models import init_chat_model
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# The first time, it is not yet in cache, so it should take longer
start_time = time.time()
result = llm.invoke("Tell me a joke")
end_time = time.time()
print(result)
print(f"Time elapsed: {end_time - start_time:.4f} seconds.")

# The second time it is, so it goes faster
start_time = time.time()
result = llm.invoke("Tell me a joke")
end_time = time.time()
print(result)
print(f"Time elapsed: {end_time - start_time:.4f} seconds.")