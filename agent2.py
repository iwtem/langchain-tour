import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool, StructuredTool
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

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

    # 提示词
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 初始化代理
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    response = agent_executor.invoke({"input": "今天是几月几日"})

    print(response)