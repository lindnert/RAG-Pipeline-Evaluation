import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from api_key import API_KEY
import langchain
import utils
import pandas as pd
from bert_score import score as bert_score


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
    base_data = df.iloc[1500:1501]

    passages_series = base_data["passages"]
    passages = passages_series.apply(lambda x: x['passage_text']).to_list()
    data = [chunk for list_of_chunks in passages for chunk in list_of_chunks]

    questions = base_data["query"].to_list()
    answers = []
    for item in base_data["answers"]:
        if len(item)>0:
            if isinstance(item, np.ndarray):
                answers.append(item[0])
            else:
                answers.append(item)
        else:
            answers.append('')

    print(f'amount of questions: {len(questions)}')
    print(questions)
    print(f'amount of answers: {len(answers)}')
    print(answers)

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

chat_model = ChatOpenAI(openai_api_key=API_KEY, verbose=False)

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
)

rouge_scores_average = {
    'rouge-1 (F-Score)': 0,
    'rouge-2 (F-Score)': 0,
    'rouge-l (F-Score)': 0
}

count = 0  # This will be used for all metrics

for index, question in enumerate(questions):
    gold_answer = answers[index]
    if gold_answer:
        system_answer = chain.invoke(question)
        rouge_scores = utils.get_rouge_scores(system_answer, gold_answer)
        for score in rouge_scores:
            rouge_scores_average[score['Metric']] += score['Result']

        bleurt_score = utils.get_bleurt_score(gold_answer, system_answer)
        print(f'BLEURT Score: {bleurt_score}')
        count += 1
        print(f'Question: {question}')
        print(f'Answer by System: {system_answer} \n\n')
        print(f'Gold Answer: {gold_answer} \n\n')
        print(f'Rouge Scores: {rouge_scores}')

for score_type in rouge_scores_average:
    rouge_scores_average[score_type] /= count
    print(f'{score_type} Average: {rouge_scores_average[score_type]}')



