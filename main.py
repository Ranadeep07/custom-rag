import google.generativeai as genai
from chunking import chunk_text
from embeddings import embed
from vectorstore import VectorStore
from rag import answer
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

text = open("data.txt").read()

chunks = chunk_text(text, 600, 100)
embeddings = embed(chunks)

store = VectorStore(len(embeddings[0]))
store.add(embeddings)

print("chunkks - ", chunks)
print("embeddings - ", embeddings)

# FastAPI setup
app = FastAPI(title="Gemini RAG API")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        result = answer(request.query, chunks, store)
        return {"query": request.query, "answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
