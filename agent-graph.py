import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

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

    system_message = "You are a helpful assistant"

    # 初始化代理
    langgraph_agent_executor = create_react_agent(llm, tools, prompt=system_message)

    query = "今天是几月几日"

    messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})

    print({
        "input": query,
        "output": messages["messages"][-1].content,
    })