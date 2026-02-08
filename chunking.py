import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def chunk_text(text, chunk_size, overlap):
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk))

    return chunks
