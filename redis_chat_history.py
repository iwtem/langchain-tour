import os

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    REDIS_URL = os.getenv("REDIS_URL")

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from Chinese into English"),
        ("user", "{text}")
    ])

    parser = StrOutputParser()

    chain = prompt_template | llm | parser

    # 初始化 RedisChatMessageHistory
    history = RedisChatMessageHistory(session_id="user_123", redis_url=REDIS_URL)

    # history.add_user_message("Hello, world!")
    # history.add_ai_message("Hello, world!")

    # runnable = RunnableWithMessageHistory(llm, get_session_history=lambda: history)
    runnable = RunnableWithMessageHistory(chain, get_session_history=lambda: history)

    runnable.invoke({ "text": "在分布式系统中，使用 Redis 存储用户会话信息，实现会话的共享和管理" })


