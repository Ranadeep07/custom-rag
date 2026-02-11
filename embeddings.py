# embeddings.py
import os
from google import genai
from google.genai import types
from config import EMBED_MODEL  # e.g., "gemini-embedding-001"

# Initialize client with API key from environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed(texts):
    """
    Embed a list of text strings using Google GenAI embeddings.
    Returns a list of embedding vectors.
    """
    embeddings = []

    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,          # e.g., "gemini-embedding-001"
            contents=[text],            # Must be a list
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"  # Supported values: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
            )
        )
        embeddings.append(response.embeddings[0].values)

    return embeddings
