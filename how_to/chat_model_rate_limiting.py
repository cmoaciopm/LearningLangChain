import os, getpass
import time

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,   # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1, # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10         # Controls the maximum burst size.
)

"""
model = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456",
    rate_limiter=rate_limiter
)
"""
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"],
    rate_limiter=rate_limiter
)

for _ in range(5):
    print(f"Start to call the model at {time.strftime('%H:%M:%S')}")
    tic = time.time()
    model.invoke("hello")
    toc = time.time()
    print(f"End the call to model at {time.strftime('%H:%M:%S')}")
    print(toc - tic)
