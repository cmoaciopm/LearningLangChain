from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

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

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert extraction algorithm.
        Only extract relevant information from the text.
        If you do not know the value of an attribute asked to extract,
        return null for the attribute's value.
    """),
    # Please see the how-to about improving performance with
    # reference examples.
    # MessagesPlaceholder('examples'),
    ("human", "{text}"),
])

llm = init_chat_model(
    model_provider="openai",
    model="qwen3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

structured_llm = llm.with_structured_output(schema=Person)
text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
result = structured_llm.invoke(prompt)
print(result)

structured_llm = llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
prompt = prompt_template.invoke({"text": text})
result = structured_llm.invoke(prompt)
print(result)