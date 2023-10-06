import requests
from fastapi import FastAPI, Request

app = FastAPI()

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
def generate():
    return {"generated_text": "Placeholder for generated text."}