from sklearn.metrics.pairwise import cosine_similarity
from numpy import array
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text1 = '我喜欢吃苹果'
text2 = '我最爱吃的水果是苹果'
text3 = '今天天气不错'

# 获取文本向量
vector1 = array(embeddings.embed_query(text1)).reshape(1, -1)
vector2 = array(embeddings.embed_query(text2)).reshape(1, -1)
vector3 = array(embeddings.embed_query(text3)).reshape(1, -1)

# 计算余弦相似度
similarity12 = cosine_similarity(vector1, vector2)[0][0]
similarity13 = cosine_similarity(vector1, vector3)[0][0]

print(similarity12, '\n')
print(similarity13)