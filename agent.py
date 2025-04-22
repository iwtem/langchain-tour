import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool, StructuredTool
from langchain.agents import initialize_agent, AgentType

load_dotenv()

@tool
def get_current_date():
    """
    获取今天的日期
    :return: str
    """
    return datetime.datetime.today().strftime("%Y-%m-%d")


if __name__ == '__main__':
    # 创建大模型
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 定义工具
    tools = [get_current_date]

    # 初始化代理
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )


    query = "今天是几月几日"

    response = agent.invoke(query)

    print(response)