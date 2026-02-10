from google import genai
from google.genai import types
from config import EMBED_MODEL

client = genai.Client()

def embed(texts):
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=types.EmbedContentConfig.TaskType.RETRIEVAL_DOCUMENT
            )
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings
