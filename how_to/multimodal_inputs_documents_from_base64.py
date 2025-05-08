import base64

from langchain_google_genai import ChatGoogleGenerativeAI

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

def pdf_to_base64(file_path):
    with open(file_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')

pdf_data = pdf_to_base64("../resources/nke-10k-2023.pdf")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe the document:"
        },
        {
            "type": "file",
            "source_type": "base64",
            "data": pdf_data,
            "mime_type": "application/pdf"
        }
    ]
}
response = llm.invoke([message])
print(response.text())