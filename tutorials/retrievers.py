import os
import getpass
import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

if not os.environ.get("QWEN_API_KEY"):
    os.environ["QWEN_API_KEY"] = getpass.getpass("Enter Qwen API key: ")

file_path = "../resources/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

"""
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.environ["QWEN_API_KEY"]
)
"""
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
print("\n")

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)



question = "How many distribution centers does Nike have in the US?"
print(f"Question: {question}")
print(f"======Searching by string query based on similarity")
results = vector_store.similarity_search(question)
print(results[0])
print("\n")

question = "When was Nike incorporated?"
print(f"Question: {question}")
print(f"======Searching by string query based on similarity asynchronously")
async def async_search(question:str):
     return await vector_store.asimilarity_search(question)
results = asyncio.run(async_search(question))
print(results[0])
print("\n")

question = "What was Nike's revenue in 2023?"
print(f"Question: {question}")
print(f"======Return the score during searching")
results = vector_store.similarity_search_with_score(question)
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)
print("\n")

question = "How were Nike's margins impacted in 2023?"
print(f"Question: {question}")
print(f"======Searching by embedded query based on similarity")
embedding = embeddings.embed_query(question)
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
print("\n")

print("======Using retriever to execute batch searching")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)
result = retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?"
])
print(result)