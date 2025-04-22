from  langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 需要先启动 redis/redis-stack 这个 docker 容器

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

redis_url = "redis://localhost:6379"

redis_config = RedisConfig(redis_url=redis_url, index_name="fruits")

vector_store = RedisVectorStore(embedding_model, config=redis_config)

vector_store.add_texts(["apple is sweet", "banana is long", "orange", "strawberry", "grape", "watermelon"])

# result = vector_store.similarity_search_with_score("甜")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
result = retriever.invoke("sweet")

print(result)


