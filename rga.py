import os

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import langchain

from api_key import API_KEY

langchain.debug = True

question = "Why do people prefer RAG over previous methods?"

# Data
data = []
number_of_docs = 0

text_files = os.listdir('Texts')
print(f'text_file_list: {text_files}')
for text_file in text_files:
    file_path = os.path.join('Texts', text_file)
    doc = TextLoader(file_path=file_path, encoding='utf-8').load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50) # chunk-size by characters
    chunks = text_splitter.split_documents(doc)
    data.extend(chunks)
    number_of_docs += 1

print(f'{number_of_docs} docs loaded.')

# Processing
"""create a vector representation of the provided texts using OpenAI embedding mode
and stores them in the FAISS index for efficient retrieval."""
vectorstore = FAISS.from_documents(
    data, embedding=OpenAIEmbeddings(openai_api_key=API_KEY)
)
print(f'Chunks in vectorstore: {vectorstore.index.ntotal}')
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(openai_api_key=API_KEY, verbose=True)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = chain.invoke(question)

print(f'Question: {question}')
print(f'Answer: {answer}')
