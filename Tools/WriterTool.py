from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# import prompt template modules
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
# taking the output of LLM and parse into desired format
from langchain_core.output_parsers import StrOutputParser
# langchain runnablePassthrough, input sth and get the output through a chain
from langchain_core.runnables import RunnablePassthrough
# langchain llm
from langchain_openai import ChatOpenAI


def write(query: str):
    """按用户要求生成文章"""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )
    # print('current template is ', template)

    chain = {"query": RunnablePassthrough()} | template | ChatOpenAI() | StrOutputParser()

    return chain.invoke(query)


if __name__ == "__main__":
    print(write("写一封邮件给张三，内容是：你好，我是李四。"))
