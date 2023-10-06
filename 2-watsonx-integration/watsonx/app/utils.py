from typing import Any, Mapping

import requests
from pydantic import BaseModel

URL = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"
API_KEY = "xxx"


class GenerateRequest(BaseModel):
    model_name: str
    model_parameters: Mapping[str, Any]
    prompt: str
    k_docs: int


class GenerateResponse(BaseModel):
    generated_text: str


def get_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    iam_token = response.json()["access_token"]
    return iam_token


def ask_question(question, db, num_docs, payload):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {get_token()}",
    }

    context = " ".join(
        [x.page_content for x in db.similarity_search(question, k=num_docs)]
    )

    prompt = f"""Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}

    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    payload.update({"input": prompt})
    r = requests.post(URL, headers=headers, json=payload)
    generated_text = r.json()["results"][0]["generated_text"]
    return generated_text
