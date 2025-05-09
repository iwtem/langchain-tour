from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text = "This is a test query."

result = embeddings.embed_query(text)

print(result)
print(len(result))