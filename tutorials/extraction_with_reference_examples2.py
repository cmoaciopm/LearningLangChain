import getpass
import os
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.utils.function_calling import tool_example_to_messages

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

class Person(BaseModel):
    """Information about a person."""

    # ^Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an 'optional' -- this allows the model to decline to extract it!
    # 2. Each field has a 'description' -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(default=None, description="Height measured in meters")

class Data(BaseModel):
    """Extracted data about People."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

examples = [
    (
        "The ocean is vast and blue. It's more that 20,000 feet deep.",
        Data(people=[])
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)])
    )
]

messages = []

for txt, tool_call in examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

for message in messages:
    message.pretty_print()

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon."
}

structured_llm = llm.with_structured_output(schema=Data)
result = structured_llm.invoke([message_no_extraction])
print(result)

result = structured_llm.invoke(messages + [message_no_extraction])
print(result)