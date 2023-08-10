from typing import Any, Dict, Iterable, List, Optional

import langchain
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


class MiniLML6V2EmbeddingFunctionLangchain(langchain.embeddings.openai.Embeddings):
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode(texts).tolist()

    def embed_query(self, query):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode([query]).tolist()[0]


def access_token(apikey):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={apikey}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    iam_token = response.json()["access_token"]
    return iam_token


def ingest_pdf(file_path):
    loader = PyPDFLoader("ED-e-KYC-2023.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(loader.load())
    db = FAISS.from_documents(texts, MiniLML6V2EmbeddingFunctionLangchain())
    return db


def make_prompt(docs, question):
    context = " ".join([doc.page_content for doc in docs])
    return (
        f"Answer the question using the context provided."
        + f"Context:\n\n"
        + f"{context}:\n\n"
        + f'If the question is unanswerable, say "unanswerable".'
        + f"Question: {question}"
    )
