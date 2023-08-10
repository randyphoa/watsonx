import time
from dataclasses import dataclass, field

import requests
import langchain
import pandas as pd
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

API_KEY = "0Rs_ysQb-gXbQO8NPRX9v8ByvHyKhGJLpBDCK5zOc50P"
PROJECT_ID = "0353fa90-88c0-44d2-b6e7-ab143db3f01d"


class MiniLML6V2EmbeddingFunctionLangchain(langchain.embeddings.openai.Embeddings):
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode(texts).tolist()

    def embed_query(self, query):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode([query]).tolist()[0]


@dataclass
class Parameters:
    chunk_size = 1000
    chunk_overlap = 10
    model = None
    temperature = 0.7
    top_k = 50
    top_p = 0.15
    repetition_penalty = 1.0
    min_new_tokens = 1
    max_new_tokens = 300
    search_k = 8


@dataclass
class Engine:
    params: Parameters = field(default_factory=Parameters)
    db: FAISS = None
    file_name: str = None

    def make_prompt(self, question):
        docs = self.db.similarity_search(question, k=self.params.search_k)
        context = " ".join([doc.page_content for doc in docs])
        prompt = (
            f"Answer the question using the context provided."
            + f"Context:\n\n"
            + f"{context}:\n\n"
            + f'If the question is unanswerable, say "unanswerable".'
            + f"Question: {question}"
        )
        return prompt, docs

    def access_token(self):
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
        response = requests.post(url, headers=headers, data=data)
        iam_token = response.json()["access_token"]
        return iam_token

    def generate(self, prompt):
        url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token()}",
        }
        payload = {
            "model_id": self.params.model,
            "input": prompt,
            "parameters": {
                "decoding_method": "sample",
                "max_new_tokens": self.params.max_new_tokens,
                "min_new_tokens": self.params.min_new_tokens,
                "random_seed": 12345,
                "stop_sequences": [],
                "temperature": self.params.temperature,
                "top_k": self.params.top_k,
                "top_p": self.params.top_p,
                "repetition_penalty": self.params.repetition_penalty,
            },
            "project_id": PROJECT_ID,
        }

        response = requests.post(url, headers=headers, json=payload)

        return response.json()["results"][0]["generated_text"]


if "engine" not in st.session_state:
    st.session_state.engine = Engine()

if "previous_query" not in st.session_state:
    st.session_state.previous_query = None

engine = st.session_state.engine

with st.sidebar:
    st.header("Settings")
    st.markdown("---")

    st.subheader("Embedding")
    engine.params.chunk_size = st.slider(
        label="Chunk size (in characters)",
        value=1000,
        min_value=500,
        max_value=2000,
        step=100,
        # disabled=(
        #     st.session_state.engine.is_file_loaded()
        #     or st.session_state.engine.is_vector_store_loaded()
        # ),
        help="The size of the text chunks that will be used to create the embeddings.",
    )
    st.write("")
    st.write("")
    st.subheader("Model")

    engine.params.model = st.selectbox(
        "Model Name",
        (
            "google/flan-ul2",
            "eleutherai/gpt-neox-20b",
            "google/flan-t5-xxl",
            "bigscience/mt0-xxl",
            "ibm/mpt-7b-instruct2",
        ),
    )

    engine.params.temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=2.0,
        value=engine.params.temperature,
        step=0.01,
        help="Control the creativity of generated text. Higher values will lead to more randomly generated outputs.",
    )
    engine.params.top_k = st.slider(
        label="Top K",
        min_value=1,
        max_value=100,
        value=engine.params.top_k,
        step=1,
        help="The number of tokens to sample from. Set the number of highest probability vocabulary tokens to keep for top-k-filtering. Lower values make it less likely the model will go off topic.",
    )
    engine.params.top_p = st.slider(
        label="Top P",
        min_value=0.0,
        max_value=1.0,
        value=engine.params.top_p,
        step=0.01,
        help="The cumulative probability of the most likely tokens to sample from. If < 1.0, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are used.",
    )
    engine.params.repetition_penalty = st.slider(
        label="Repetition penalty",
        min_value=1.0,
        max_value=2.0,
        value=engine.params.repetition_penalty,
        step=0.01,
        help="The penalty for repeating tokens. 1.00 means no penalty.",
    )
    cols = st.columns(2)
    with cols[0]:
        engine.params.min_new_tokens = st.number_input(
            label="Min new tokens",
            value=engine.params.min_new_tokens,
            min_value=1,
            max_value=1000,
            step=1,
            help="The minimum number of tokens that will be added to the previous answer.",
        )
    with cols[1]:
        engine.params.max_new_tokens = st.number_input(
            label="Max new tokens",
            value=engine.params.max_new_tokens,
            min_value=1,
            max_value=1000,
            step=1,
            help="The maximum number of tokens that will be added to the previous answer.",
        )


st.header("Retrevial Augmented Generation")

st.markdown("---")

if not engine.file_name:
    file = st.file_uploader("Upload a PDF file from your computer", type="pdf")
    if file is not None:
        progress_bar = st.progress(0, text="Uploading file ...")
        with open(file.name, mode="wb") as f:
            f.write(file.getvalue())
        time.sleep(2)

        progress_bar.progress(25, text="Chunking file ...")
        loader = PyPDFLoader(file.name)

        progress_bar.progress(50, text="Chunking file ...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=10
        )
        texts = text_splitter.split_documents(loader.load())

        progress_bar.progress(75, text="Chunking file ...")
        engine.db = FAISS.from_documents(texts, MiniLML6V2EmbeddingFunctionLangchain())

        engine.file_name = file.name
        progress_bar.progress(100, text=f"File {file.name} is successfully saved.")
        time.sleep(2)
        progress_bar.empty()
        st.experimental_rerun()
else:
    query = st.text_area(
        label="Ask your question",
        value="",
        placeholder="",
        height=200,
    )
    if query and (
        query != st.session_state.previous_query or st.session_state.rerun_requested
    ):
        with st.spinner("Searching..."):
            st.markdown("")
            st.session_state.rerun_requested = False
            st.session_state.previous_query = query

            prompt, docs = engine.make_prompt(question=query)
            answer = engine.generate(prompt)
            st.write(answer)
            # if st.session_state.engine.config.debug:
            #     st.markdown("")
            #     st.subheader("Debug")
            #     st.write(docs)
