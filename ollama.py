from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following from Chinese into English"),
    ("user", "{text}")
])

chain = prompt | model | parser

result = chain.invoke("你好，世界")

print(result)
