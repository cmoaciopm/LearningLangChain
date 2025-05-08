from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, filter_messages
from openai import api_key

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice")
]

filtered_messages = filter_messages(messages, include_types="human")
print(filtered_messages)

filtered_messages = filter_messages(messages, exclude_names=["example_user", "example_assistant"])
print(filtered_messages)

filtered_messages = filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])
print(filtered_messages)

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456",
    temperature=0
)

filter_ = filter_messages(exclude_names=["example_user", "example_assistant"])
chain = filter_ | llm
result = chain.invoke(messages)
print(result)

print(filter_.invoke(messages))