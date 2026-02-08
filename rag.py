import google.generativeai as genai
from config import GEN_MODEL, TOP_K, EMBED_MODEL
from embeddings import embed

def answer(query, chunks, store):
    q_embed = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )["embedding"]

    indices = store.search(q_embed, TOP_K)
    context = "\n\n".join([chunks[i] for i in indices])

    prompt = f"""
Answer using ONLY the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    model = genai.GenerativeModel(GEN_MODEL)
    return model.generate_content(prompt).text
