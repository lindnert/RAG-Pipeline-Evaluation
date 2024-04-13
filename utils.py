import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from rouge import Rouge
from bleurt import score as bleurt_score


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

def cache_embeddings(embeddings):
    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )
    return cached_embedder

def get_rouge_scores(text1: str, text2: str):
    if not text1 and text2:
        return
    rouge = Rouge()
    rouge_scores_out = []
    rouge_scores_raw = rouge.get_scores(text1, text2)
    for metric in ["rouge-1", "rouge-2", "rouge-l"]:
        for label in ["F-Score"]:
            eval_score = rouge_scores_raw[0][metric][label[0].lower()]
            rouge_scores_out.append({"Metric": f"{metric} ({label})", "Result": eval_score,})
    return rouge_scores_out

def get_bleurt_score(gold_answer: str, system_answer: str):
    checkpoint = "bleurt/BLEURT-20"
    bleurt_scorer = bleurt_score.BleurtScorer(checkpoint)
    return bleurt_scorer.score(references=[gold_answer], candidates=[system_answer])