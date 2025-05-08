import base64
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

prompt = ChatPromptTemplate([
    {"role": "system", "content": "Describe the image provided."},
    {"role": "user", "content": [{"type": "image", "source_type": "url", "url": "{image_url}"}]}
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

chain = prompt | llm
response = chain.invoke({"image_url": url})
print(response.text())