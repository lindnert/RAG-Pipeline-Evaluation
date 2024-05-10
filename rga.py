import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from API_keys.openai_api_key import OPENAI_API_KEY
from API_keys.mistral_api_key import MISTRAL_API_KEY
from API_keys.llama_api_key import LLAMA_API_KEY
import utils
import pandas as pd
from bert_score import score as bert_score
from bleurt import score as bleurt_score
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import time

#from datasets import load_dataset

dataset_options = ["ms_marco"]
model_options = ["openai", "mistral", "llama"]
prompt_long = """"""
prompt_short = """Answer the question based only on the following context:
{context}

Question: {question}
"""

#langchain.debug = True
data = None
embedding = None
vectorstore = None
chat_model = None
questions = []
answers = []

try:
    with open("evaluation_results.txt", "w") as file:
        file.write("Evaluation results:\n\n")
except IOError as e:
    print(f"An error occurred while writing to the file: {e}")

for dataset in dataset_options:
    dataset_to_use = dataset

    start_time = time.time()

    # Data
    if dataset_to_use == "ms_marco":
        # data = load_dataset(dataset_name="ms_marco", version="v2.1", split="test")
        df = pd.read_parquet('0000.parquet')
        base_data = df.iloc[1800:1805]
        base_data_length = len(base_data)

        passages_series = base_data["passages"]
        passages = passages_series.apply(lambda x: x['passage_text']).to_list()
        data = list(set([chunk for list_of_chunks in passages for chunk in list_of_chunks]))

        questions = base_data["query"].to_list()
        for item in base_data["answers"]:
            if len(item) > 0:
                if isinstance(item, np.ndarray):
                    answers.append(item[0])
                else:
                    answers.append(item)
            else:
                answers.append('')

        print(f'amount of questions: {len(questions)}')
        print(questions)
        print(f'amount of gold answers: {len(answers)}')
        print(answers)

        print(f"Elapsed time for loading data: {time.time() - start_time} seconds")

    for model in model_options:
        model_to_use = model
        start_time = time.time()

        if model_to_use == "openai":
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
            chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, verbose=False)
            embedding_to_use, chat_model_to_use = model_to_use, model_to_use

        elif model_to_use == "mistral":
            embedding = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
            chat_model = ChatMistralAI(api_key=MISTRAL_API_KEY, temperature=0)
            embedding_to_use, chat_model_to_use = model_to_use, model_to_use

        elif model_to_use == "llama":
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
            llama = LlamaAPI(LLAMA_API_KEY)
            chat_model = ChatLlamaAPI(client=llama)
            embedding_to_use = "openai"
            chat_model_to_use = model_to_use

        cached_embedding = utils.cache_embeddings(embedding)
        print(f"Embedding model: {embedding_to_use}\n")
        print(f"Chat model: {chat_model_to_use}\n")

        # Processing
        """create a vector representation of the provided texts using OpenAI embedding mode
        and stores them in the FAISS index for efficient retrieval."""
        vectorstore = FAISS.from_texts(data, embedding=cached_embedding)

        print(f'Chunks in vectorstore: {vectorstore.index.ntotal}')
        retriever = vectorstore.as_retriever(k=4)

        prompt_template = """Answer the question briefly, precisely and concisely \
        based only the context provided to you. \
        If the information required to answer the question is not contained in the provided context, \
        return an empty string.
                
        Context:
        {context}
        
        Question:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | chat_model
                | StrOutputParser()
        )

        print(f"Elapsed time for loading model: {time.time() - start_time} seconds")

        rouge_avg = {
            'rouge-1 (F-Score)': 0,
            'rouge-2 (F-Score)': 0,
            'rouge-l (F-Score)': 0
        }

        bleurt_avg = []
        checkpoint = "bleurt/BLEURT-20"
        bleurt_scorer = bleurt_score.BleurtScorer(checkpoint)
        bert_avg = {"bert_P": 0, "bert_R": 0, "bert_F1": 0}
        meteor_avg = []
        sas_avg = []

        count = 0  # This will be used for all metrics

        for index, question in enumerate(questions):
            gold_answer = answers[index]
            if gold_answer:
                start_time = time.time()
                system_answer = chain.invoke(question)
                print(f"Elapsed time for creating system answer: {time.time() - start_time} seconds")

                start_time = time.time()
                rouge_scores = utils.get_rouge_scores(system_answer, gold_answer)
                for score in rouge_scores:
                    rouge_avg[score['Metric']] += score['Result']

                bleurt_results = utils.get_bleurt_score(bleurt_scorer, gold_answer, system_answer)
                bleurt_avg.extend(bleurt_results)

                bert_P, bert_R, bert_F1 = bert_score([system_answer], [gold_answer], lang="en", model_type="bert-base-uncased")
                bert_P, bert_R, bert_F1 = bert_P.item(), bert_R.item(), bert_F1.item()
                bert_avg["bert_P"] += bert_P
                bert_avg["bert_R"] += bert_R
                bert_avg["bert_F1"] += bert_F1

                sas_score = utils.calculate_sas(system_answer, gold_answer)
                sas_avg.extend(sas_score)

                # Tokenize for METEOR
                gold_answer_tok = word_tokenize(gold_answer)
                system_answer_tok = word_tokenize(system_answer)
                meteor_score_result = meteor_score.single_meteor_score(gold_answer_tok, system_answer_tok)
                meteor_avg.append(meteor_score_result)

                count += 1
                print(f'Question: {question}')
                print(f'System Answer: {system_answer}')
                print(f'Gold Answer: {gold_answer}\n')
                print(f'Rouge Scores: {rouge_scores}')
                print(f'BLEURT Score: {bleurt_results}')
                print(f'BERT Precision, Recall and F1: {bert_P, bert_R, bert_F1}')
                print(f'METEOR Score: {meteor_score_result}')
                print(f'SAS Score: {sas_score}')
                print('\n----------------------------------\n')

                print(f"Elapsed time for creating metrics: {time.time() - start_time} seconds")
                start_time = time.time()

        try:
            with open("evaluation_results.txt", "a") as file:
                file.write(f"Dataset and Size: {dataset_to_use} ({base_data_length})\n")
                file.write(f"Embedding model: {embedding_to_use}\n")
                file.write(f"Chat model: {chat_model_to_use}\n")
                for score_type in rouge_avg:
                    rouge_avg[score_type] /= count
                    file.write(f'{score_type} Average: {rouge_avg[score_type]}\n')

                bleurt_score_avg = sum(bleurt_avg) / count
                file.write(f'BLEURT Score Average: {bleurt_score_avg}\n')

                for score_type in bert_avg:
                    bert_avg[score_type] /= count
                    file.write(f'{score_type} Average: {bert_avg[score_type]}\n')

                meteor_avg = sum(meteor_avg) / count
                file.write(f'METEOR Score Average: {meteor_avg}\n')

                sas_avg = sum(sas_avg) / count
                file.write(f'SAS Score: {sas_avg}\n')

                file.write('\n\n')

            print(f"Results saved to evaluation_results.txt")
            print('\n\n----------------------------------\n----------------------------------\n\n')

            print(f"Elapsed time for writing results: {time.time() - start_time} seconds")

        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
