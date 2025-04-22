import os
import shutil

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnableWithMessageHistory
from langchain_core.utils import from_env
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

load_dotenv()

if __name__ == '__main__':
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    openAI = ChatOpenAI(model="gpt-4o-mini")
    deepseek = ChatDeepSeek(model="deepseek-chat")

    messages = [
        # SystemMessage("Translate the following from Chinese into English"),
        HumanMessage("介绍一下你自己")
    ]

    prompt_template_en = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from Chinese into English"),
        ("user", "{text}")
    ])

    prompt_template_fr = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from Chinese into French"),
        ("user", "{text}")
    ])

    parser = StrOutputParser()

    chain_en = prompt_template_en | openAI | parser
    chain_fr = prompt_template_fr | openAI | parser

    parallel_chains = RunnableMap({
        "en_translation": chain_en,
        "fr_translation": chain_fr
    })

    final_chain = parallel_chains | RunnableLambda(lambda x: f"English: {x['en_translation']}\nFrench:{x['fr_translation']}")

    result = final_chain.invoke({ "text": "很高兴认识你" })
    # result = openAI.stream(messages)

    # for chunk in result:
    #     print(chunk.text())

    print(result)
    # print(from_env('LANGSMITH_API_KEY'))

    final_chain.get_graph().print_ascii()
