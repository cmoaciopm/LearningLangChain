from langchain_openai import ChatOpenAI
import logging
logging.basicConfig(level=logging.DEBUG)

import httpx
def log_request(request):
    print(f"Request: {request.method} {request.url}")
    print("Headers:", request.headers)
    print("Body:", request.content.decode())

def log_response(response):
    response.read()
    print(f"Response: {response.status_code}")
    print("Headers:", response.headers)
    print("Body:", response.text)

client = httpx.Client(
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    }
)

llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456",
    http_client=client
)
response = llm.invoke("What are you?")