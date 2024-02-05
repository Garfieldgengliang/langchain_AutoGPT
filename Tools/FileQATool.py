# from typing import List
# from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
# langchain document loader and split into chunks
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# langchain vector database and information retriever
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# langchain embedding models, convenient to get string embedding
from langchain.embeddings.openai import OpenAIEmbeddings
# langchain llm
from langchain.llms import OpenAI

class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")

def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]

# def load_docs(filename: str) -> List[Document]:
def load_docs(filename: str):
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split() # load Documents and split into chunks. Chunks are returned as Documents.
    return pages

def ask_docment(filename: str,  query: str,) -> str:
    """根据一个PDF文档的内容，回答一个问题"""
    raw_docs = load_docs(filename) # use langchain file loader to load files
    if len(raw_docs) == 0:
        return "抱歉，文档内容为空"
    text_splitter = RecursiveCharacterTextSplitter( # split the raw file into chunks
                        chunk_size=200,
                        chunk_overlap=100,
                        length_function=len,
                        add_start_index=True,
                    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"
    # build Chroma vector database, which will be used later to retrieve information given index
    db = Chroma.from_documents(documents, OpenAIEmbeddings(model="text-embedding-ada-002"))
    # langchain QA chain,
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(
            temperature=0,
            model_kwargs={
                "seed": 42
            },
        ),  # 语言模型
        chain_type="stuff",  # prompt的组织方式，后面细讲
        retriever=db.as_retriever()  # 检索器
    )
    response = qa_chain.run(query+"(请用中文回答)")
    return response


if __name__ == "__main__":
    filename = "../data/供应商资格要求.pdf"
    query = "供应商流动资金有什么要求？"
    response = ask_docment(filename, query)
    print(response)