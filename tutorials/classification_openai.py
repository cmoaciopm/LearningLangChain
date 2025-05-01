import getpass
import os
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

tagging_prompt = ChatPromptTemplate.from_template(
    """
        Extract the desired information from the following passage.
        Only extract the properties mentioned in the 'Classification' function.
        Passage:
        {input}
    """
)

class Classification(BaseModel):
    sentiment: Literal["happy", "neutral", "sad"] = Field(
        description="The sentiment of the text"
    )
    language: Literal['spanish', "english", "french", "german", "italian"] = Field(
        description="The language the text is written in"
    )
    aggressiveness: Literal[1, 2, 3, 4, 5] = Field(
       description="describes how aggressive the statement is, the higher the number the more aggressive"
    )

structured_llm = llm.with_structured_output(Classification)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)
print(response.model_dump())

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)
print(response.model_dump())

inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
response = structured_llm.invoke(prompt)
print(response.model_dump())

