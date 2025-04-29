import getpass
import os

from langchain_core.prompts import ChatPromptTemplate

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

prompt_value = prompt_template.invoke({
    "language": "Italian",
    "text": "hi!"
})
print(prompt_value)

response = model.invoke(prompt_value)
print(response.content)