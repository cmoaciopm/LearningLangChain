import getpass
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

file_path = "../resources/layout-parser-paper.pdf"

loader = PyPDFLoader(file_path)
pages = []
for page in loader.lazy_load():
    pages.append(page)

print(f"{pages[0].metadata}\n")
print(pages[0].page_content)

print("=========")

vector_store = InMemoryVectorStore.from_documents(
    pages,
    DashScopeEmbeddings()
)
docs = vector_store.similarity_search("What is LayoutParser?", k=2)
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')