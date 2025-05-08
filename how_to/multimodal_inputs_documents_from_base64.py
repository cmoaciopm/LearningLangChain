import base64
import getpass
import os

from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

def pdf_to_base64(file_path):
    with open(file_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_data = pdf_to_base64("../resources/nke-10k-2023.pdf")


# TODO So far, I don't find a free provider which accepts PDF documents.
llm = init_chat_model(
    model="qwen3:8b",
    model_provider="openai",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)
"""
llm = init_chat_model(
    model="qwen-vl-max-latest",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)
"""
message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:"
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf"
        }
    ]
}
response = llm.invoke([message])
print(response.text())