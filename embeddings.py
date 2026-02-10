from google import genai
from google.genai import types
from config import EMBED_MODEL

client = genai.Client()

def embed(texts):
    """
    Embed a list of text strings using Google GenAI embeddings.
    Returns a list of embeddings.
    """
    embeddings = []

    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[text],  # ✅ must be a list
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"  # ✅ string, not enum
            )
        )
        embeddings.append(response.embeddings[0].values)

    return embeddings
