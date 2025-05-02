from typing import Optional, Union

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline to the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""

    response: Annotated[str, ..., "A conversational response to the user's query"]

class FinalResponse(TypedDict):
    final_output: Union[Joke, ConversationalResponse]

structured_llm = llm.with_structured_output(FinalResponse)
result = structured_llm.invoke("Tell me a joke about cats")
print(result)

result = structured_llm.invoke("How are you today?")
print(result)