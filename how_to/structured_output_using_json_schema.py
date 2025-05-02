from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke"
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke"
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None
        }
    },
    "required": ["setup", "punchline"]
}

structured_llm = llm.with_structured_output(json_schema)
result = structured_llm.invoke("Tell me a joke about cats")
print(result)

# If output type is a dict(either TypedDict class or JSON Schema dict),
# we can use stream output from the structured model.
for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)