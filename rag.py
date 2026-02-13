from google import genai
from google.genai import types
from config import GEN_MODEL, TOP_K, EMBED_MODEL

client = genai.Client()

def answer(query, chunks, store):
    # Embed the query
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    q_embed = response.embeddings[0].values

    # Vector search
    indices = store.search(q_embed, TOP_K)

    print("\n========== TOP K RETRIEVED CHUNKS ==========")
    for rank, idx in enumerate(indices):
        print(f"\nRank {rank + 1} | Chunk Index: {idx}")
        print("--------------------------------------------")
        print(chunks[idx])
    print("============================================\n")
    
    context = "\n\n".join([chunks[i] for i in indices])

    prompt = f"""
Answer using ONLY the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    # Generate answer
    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
    )

    return response.text
