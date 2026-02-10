from google import genai
from config import EMBED_MODEL

client = genai.Client()

def embed(texts):
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            task_type="retrieval_document",
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings
