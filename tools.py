import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool, StructuredTool

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

    # 绑定大模型工具
    llm_with_tools = llm.bind_tools([get_current_date])

    query = "今天是几月几日"

    messages = [query]

    # 询问大模型，大模型会判断需要调用工具，然后返回一个工具调用的请求
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    print(ai_msg, '\n')

    all_tools = { "get_current_date": get_current_date }

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            selected_tool = all_tools[tool_call["name"].lower()]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg, '\n')
            messages.append(tool_msg)

    msg = llm_with_tools.invoke(messages)
    print(msg)