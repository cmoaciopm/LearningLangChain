from typing import Optional

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

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline to the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke("Tell me a joke about cats")
print(result)

# If output type is a dict(either TypedDict class or JSON Schema dict),
# we can use stream output from the structured model.
for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)