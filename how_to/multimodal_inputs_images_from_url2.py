import getpass
import os

from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

#image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_url = "https://q3.itc.cn/q_70/images03/20240608/cbbeb83af3cf49c5b48ee65f19710d8c.jpeg"

llm = init_chat_model(
    model="qwen-vl-max-latest",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Are these two images the same?"},
        {"type": "image", "source_type": "url", "url": image_url},
        {"type": "image", "source_type": "url", "url": image_url}
    ]
}
response = llm.invoke([message])
print(response.text())