from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

model = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

chunks = []
for chunk in model.stream("What color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)