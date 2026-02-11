import os
from google import genai
from google.genai import types
from config import EMBED_MODEL

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]

    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return [e.values for e in response.embeddings]