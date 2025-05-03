from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

model = init_chat_model(
    model_provider="openai",
    model="llama3:8b",
    base_url="http://localhost:11434/v1",
    api_key="123456"
)

chain = model | JsonOutputParser()

for text in chain.stream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(text, flush=True)