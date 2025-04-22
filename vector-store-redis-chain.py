from  langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 需要先启动 redis/redis-stack 这个 docker 容器

load_dotenv()

def format_prompt_value(text):
    return text.to_string()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

redis_url = "redis://localhost:6379"

redis_config = RedisConfig(redis_url=redis_url, index_name="fruits")

vector_store = RedisVectorStore(embedding_model, redis_config=redis_config)

prompt = ChatPromptTemplate.from_messages([
    ("human", "{text}")
])

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chain = prompt | format_prompt_value | retriever

result = chain.invoke({ "text": "what is the fruit that is sweet" })

print(result)


