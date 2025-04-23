from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader

file_path = "./example_data/836a8c3f35be444cb87d504d1c2be85a.pdf"

# loader = PDFPlumberLoader(file_path)
loader = PyPDFLoader(file_path)

docs = loader.load()

print(docs[0].metadata)
print(docs[0].page_content)
print(len(docs[0].page_content))