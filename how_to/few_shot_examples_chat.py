import os, getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

result = model.invoke("What is 2 🦜 9?")
print(result)

print("=========")

examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples = examples
)

print(few_shot_prompt.invoke({}).to_messages())

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a wondrous wizard of math."),
    few_shot_prompt,
    ("human", "{input}")
])

chain = final_prompt | model

result = chain.invoke({"input": "What is 2 🦜 9?"})
print(result)