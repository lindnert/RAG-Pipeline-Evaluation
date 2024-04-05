from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from api_key import API_KEY
import langchain
import utils
import pandas as pd
#from datasets import load_dataset

#langchain.debug = True
data = None
questions = []

data_to_use = "ms_marco"
embedding_to_use = "openai"

if embedding_to_use == "openai":
    embedding = OpenAIEmbeddings(openai_api_key=API_KEY, model="text-embedding-3-small")
    cached_embedding = utils.cache_embeddings(embedding)

# Data
if data_to_use == "ms_marco":
    # data = load_dataset(dataset_name="ms_marco", version="v2.1", split="test")
    df = pd.read_parquet('0000.parquet')
    base_data = df.iloc[1000:1201]

    passages_series = base_data["passages"]
    passages = passages_series.apply(lambda x: x['passage_text']).to_list()
    data = [chunk for list_of_chunks in passages for chunk in list_of_chunks]

    questions = base_data["query"].to_list()

    # Processing
    """create a vector representation of the provided texts using OpenAI embedding mode
    and stores them in the FAISS index for efficient retrieval."""
    vectorstore = FAISS.from_texts(data, embedding=cached_embedding)

elif data_to_use == "local":
    questions = ["How can RAG pipelines be evaluated?"]
    docs = utils.load_from_local()
    data = utils.chunk(docs)

    # Processing
    """create a vector representation of the provided texts using OpenAI embedding mode
    and stores them in the FAISS index for efficient retrieval."""
    vectorstore = FAISS.from_documents(data, embedding=cached_embedding)

print(f'Chunks in vectorstore: {vectorstore.index.ntotal}')
retriever = vectorstore.as_retriever(k=4)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chat_model = ChatOpenAI(openai_api_key=API_KEY)
#chat_model = ChatOpenAI(openai_api_key=API_KEY, verbose=True)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
)

for question in questions:
    answer = chain.invoke(question)
    print(f'Question: {question}')
    print(f'Answer: {answer} \n\n')



