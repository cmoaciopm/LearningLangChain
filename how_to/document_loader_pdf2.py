from langchain_unstructured import UnstructuredLoader

file_path = "../resources/layout-parser-paper.pdf"

loader_local = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res"
)
docs_local = []
for doc in loader_local.lazy_load():
    docs_local.append(doc)

print(len(docs_local))

first_page_docs = [doc for doc in docs_local if doc.metadata.get("page_number") == 1]
for doc in first_page_docs:
    print("=========")
    print(doc.page_content)