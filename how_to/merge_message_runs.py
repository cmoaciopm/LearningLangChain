import os, getpass
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, merge_message_runs
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you always respond with a joke."),
    HumanMessage([{"type": "text", "text": "i wonder why it's called langchain"}]),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage('Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'),
    AIMessage("Why, he's probably chasing after the last cup of coffee in the office!")
]

merged = merge_message_runs(messages)
print("\n\n".join([repr(x) for x in merged]))

"""
# Ollama returns error for the merged messages because the merged HumanMessage are treated as invalid message
llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456",
    temperature=0
)
"""
"""
# qwen returns error for the merged messages because the merged HumanMessage are treated as invalid message
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

merger = merge_message_runs()
chain = merger | llm
result = chain.invoke(messages)
print(result)

prompt = ChatPromptTemplate([
    ("system", "You're great at {skill}"),
    ("system", "You're also great at explaining things"),
    ("human", "{query}")
])
chain = prompt | merger | llm
result = chain.invoke({"skill": "math", "query": "what's the definition of a convergent series"})
print(result)