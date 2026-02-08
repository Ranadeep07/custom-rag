import google.generativeai as genai
from config import EMBED_MODEL

def embed(texts):
    embeddings = []
    for text in texts:
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(res["embedding"])
    return embeddings
