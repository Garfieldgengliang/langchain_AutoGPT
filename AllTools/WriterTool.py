from dotenv import load_dotenv
load_dotenv("my_api_keys.env")
import openai
import os
openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = "sk-A9DzKnDJzugLuaNqoqQ19MMANXWHaiCl8L7BjpiegLvH6IEj"

from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

writer_chain = LLMChain(llm=ChatOpenAI(), prompt=template)

if __name__ == "__main__":
    print(writer_chain.run("写一封邮件给张三，内容是：你好，我是李四。"))