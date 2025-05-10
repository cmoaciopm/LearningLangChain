import os, getpass
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

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

result = model.invoke("What is 3 🦜 3?")
print(result)

print("=========")

examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?"
    }
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = DashScopeEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human and 1 AI
    example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
)

print(few_shot_prompt.invoke(input="What's 3 🦜 3?").to_messages())

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a wondrous wizard of math."),
    few_shot_prompt,
    ("human", "{input}")
])

print(few_shot_prompt.invoke(input="What's 3 🦜 3?"))

chain = final_prompt | model
result = chain.invoke({"input": "What's 3 🦜 3?"})
print(result)