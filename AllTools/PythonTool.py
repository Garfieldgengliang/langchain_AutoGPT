import re
from dotenv import load_dotenv
load_dotenv("my_api_keys.env")
import openai
import os
openai.api_base = "https://api.fe8.cn/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')

from langchain.tools import StructuredTool
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PythonExecUtil import execute_python_code
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from AllTools.ExcelTool import get_first_n_rows, get_column_names

 # 从环境变量中加载 API keys，必须在所有 import 之前

class ExcelAnalyser:

    def __init__(self, prompts_path):
        self.prompt = PromptTemplateBuilder(prompts_path, "excel_analyser.templ").build()

    def analyse(self, query, filename):

        """分析一个结构化文件（例如excel文件）的内容。"""

        columns = get_column_names(filename)
        inspections = get_first_n_rows(filename, 3)

        chain = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=self.prompt)
        response = chain.run(
            query=query,
            filename=filename,
            columns=columns,
            inspections=inspections
        )

        #print("\n"+response+"\n")

        code = self._extract_python_code(response)

        if code:
            ans = execute_python_code(code)
            return ans
        else:
            return "没有找到可执行的Python代码"

    def _remove_marked_lines(self, input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]

        ans = '\n'.join(lines)
        return ans

    def _extract_python_code(self, text: str) -> str:
        # 使用正则表达式找到所有的Python代码块
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        # 从re返回结果提取出Python代码文本
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
            python_code = self._remove_marked_lines(python_code)
        return python_code

    def as_tool(self):
        return StructuredTool.from_function(
            func=self.analyse,
            name="AnalyseExcel",
            description="通过程序脚本分析一个结构化文件（例如excel文件）的内容。输人中必须包含文件的完整路径和具体分析方式和分析依据，阈值常量等。如果输入信息不完整，你可以拒绝回答。",
        )

if __name__ == "__main__":
    print(ExcelAnalyser(
        prompts_path="F://Langchain_AutoGPT//prompts//"
    ).analyse(
        query="8月销售额",
        filename="F://Langchain_AutoGPT//data//2023年8月-9月销售记录.xlsx"
    ))

