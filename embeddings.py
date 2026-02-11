import os
from google import genai
from config import EMBED_MODEL

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts
    )

    return [e.values for e in response.embeddings]
