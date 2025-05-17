import os
import sys
import json
import faiss

from loguru import logger
from slowapi import Limiter
from dotenv import load_dotenv
from slowapi.util import get_remote_address
from fastapi.security import APIKeyHeader
from fastapi import FastAPI, HTTPException, Request, Depends
from sentence_transformers import SentenceTransformer

# setting up and customizing loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD hh:mm:ss A}</green> | <level>{level: <8}</level> | <cyan>{file}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)
# initializing the app
load_dotenv()
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
# setting up rate limiter
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
app.state.limiter = limiter
# creating api key header scheme
header_scheme = APIKeyHeader(name="x-api-key")
API_KEY = os.getenv("API_KEY")

with open(os.path.join("data", "courses.json"), "r") as f:
    courses = json.load(f)

index = faiss.read_index(os.path.join("data", "course_index.faiss"))


@app.get("/search")
@limiter.limit("100/hour")
async def search(request: Request, q: str, api_key: str = Depends(header_scheme)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    query_embedding = model.encode([q]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, 10)

    return {"results": [courses[i] for i in indices[0].tolist()]}
