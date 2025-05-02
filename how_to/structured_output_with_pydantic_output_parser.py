import getpass
import os
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

llm = init_chat_model(
    model_provider="openai",
    model="qwen-plus-latest",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ["QWEN_API_KEY"]
)

class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(..., description="The height of the person expressed in meters.")

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]

parser = PydanticOutputParser(pydantic_object=People)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user query. Wrap the output in `json` tags\n{format_instructions}"),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

query = "Anna is 23 years old and she is 6 feet tall"
print(prompt.invoke({"query": query}).to_string())

chain = prompt | llm | parser
result = chain.invoke({"query": query})
print(result)