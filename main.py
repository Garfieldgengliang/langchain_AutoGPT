# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from AutoAgent.AutoGPT import AutoGPT
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain.agents.agent_toolkits import FileManagementToolkit


def launch_agent(agent: AutoGPT):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"

    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, verbose=True)
        print(f"{ai_icon}：{reply}\n")


def main():

    # 语言模型
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        temperature=0,
        model_kwargs={
            "seed": 42
        },
    )

    # 存储长时记忆的向量数据库
    db = Chroma.from_documents([Document(page_content="")], OpenAIEmbeddings(model="text-embedding-ada-002"))
    retriever = db.as_retriever(search_kwargs=dict(k=1))

    # 自定义工具集
    tools = [
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
        directory_inspection_tool,
        finish_placeholder,
        ExcelAnalyser(
            prompts_path="./prompts/tools",
            prompt_file="excel_analyser.json",
            verbose=True
        ).as_tool()
    ]

    # 定义智能体
    agent = AutoGPT(
        llm=llm,
        prompts_path="./prompts/main",
        tools=tools,
        work_dir="./data",
        main_prompt_file="main.json",
        final_prompt_file="final_step.json",
        max_thought_steps=20,
        memery_retriever=retriever
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
