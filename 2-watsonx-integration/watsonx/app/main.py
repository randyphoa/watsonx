import requests
import langchain

from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from .utils import GenerateRequest, GenerateResponse, ask_question

from fastapi import FastAPI, Request

API_KEY = "xxx"


class MiniLML6V2EmbeddingFunctionLangchain(langchain.embeddings.openai.Embeddings):
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode(texts).tolist()

    def embed_query(self, query):
        return MiniLML6V2EmbeddingFunctionLangchain.MODEL.encode([query]).tolist()[0]


app = FastAPI()

db = FAISS.load_local("db", MiniLML6V2EmbeddingFunctionLangchain())


@app.get("/openapi")
def openapi(request: Request):
    url = request.base_url._url[:-1]
    openapi = requests.get(f"{url}/openapi.json").json()
    openapi["openapi"] = "3.0.3"
    openapi["info"] = {
        "title": "watsonx.ai generation API endpoint",
        "version": "0.1.0",
    }
    openapi["servers"] = [{"url": url, "description": "watsonx.ai endpoint"}]
    # if "paths" in openapi:
    #     del openapi["paths"]["/"]
    if "/openapi" in openapi["paths"]:
        del openapi["paths"]["/openapi"]
    if "components" in openapi:
        del openapi["components"]["schemas"]["HTTPValidationError"]
        del openapi["components"]["schemas"]["ValidationError"]
    for k in openapi["paths"].keys():
        if "post" in openapi["paths"][k]:
            del openapi["paths"][k]["post"]["responses"]["422"]
    return openapi


@app.get("/")
def hello():
    return {"generated_text": "Hello World!"}


@app.get("/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    k_docs = request.k_docs
    prompt = request.prompt
    generated_text = ask_question(question=prompt, db=db, num_docs=k_docs)
    return {"generated_text": generated_text}
