import base64
import getpass
import os

from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_data = image_to_base64("../resources/Gfp-wisconsin-madison-the-nature-boardwalk.jpg")

llm = init_chat_model(
    model="qwen-vl-max-latest",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the weather in this image:"
        },
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": "image/jpeg"
        }
    ]
}
response = llm.invoke([message])
print(response.text())