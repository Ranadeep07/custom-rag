import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

from chunking import chunk_text
from embeddings import embed
from vectorstore import VectorStore
from rag import answer

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")  # <- token for protection

if not GEMINI_API_KEY or not API_TOKEN:
    raise ValueError("Missing GEMINI_API_KEY or API_TOKEN in environment")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Load your document and prepare embeddings
with open("data.txt") as f:
    text = f.read()

chunks = chunk_text(text, 600, 100)
embeddings = embed(chunks)

store = VectorStore(len(embeddings[0]))
store.add(embeddings)

print("Chunks:", chunks)
print("Embeddings:", embeddings)

# FastAPI app
app = FastAPI(title="Protected Gemini RAG API")

# Request model
class QueryRequest(BaseModel):
    query: str

# Endpoint with API token protection
@app.post("/ask")
def ask_question(request: QueryRequest, x_api_key: str = Header(...)):
    # Check token
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API Token")

    # Run RAG
    try:
        result = answer(request.query, chunks, store)
        return {
            "query": request.query,
            "answer": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
