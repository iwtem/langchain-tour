from  langchain_redis import RedisConfig, RedisVectorStore

redis_url = "redis://localhost:6379"

redis_config = RedisConfig(redis_url=redis_url, index_name="fruits")

vector_store = RedisVectorStore(redis_config=redis_config)








