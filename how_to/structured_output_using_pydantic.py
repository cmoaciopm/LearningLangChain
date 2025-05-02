from typing import Optional

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(default=None, description="How funny the joke is, from 1 to 10")

structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke("Tell me a joke about cats")
print(result)