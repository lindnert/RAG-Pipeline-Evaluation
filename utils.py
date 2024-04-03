import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_from_local() -> list:
    docs = []
    text_files = os.listdir('Texts')
    print(f'text_file_list: {text_files}')
    for text_file in text_files:
        file_path = os.path.join('Texts', text_file)
        doc = TextLoader(file_path=file_path, encoding='utf-8').load()
        docs.extend(doc)
    print(f'{len(docs)} docs loaded.')
    return docs

def chunk(docs: list) -> list:
    chunked_docs =[]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # chunk-size by characters
    for doc in docs:
        chunks = text_splitter.split_documents(doc)
        chunked_docs.extend(chunks)
    return chunked_docs