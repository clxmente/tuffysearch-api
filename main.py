import os
import sys

from loguru import logger
from slowapi import Limiter
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from slowapi.util import get_remote_address
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import FastAPI, HTTPException, Request, Depends

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
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# setting up rate limiter
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
app.state.limiter = limiter
# creating api key header scheme
header_scheme = APIKeyHeader(name="x-api-key")
API_KEY = os.getenv("API_KEY")
# load the faiss index from disk into memory
db = FAISS.load_local(
    os.path.join("data", "faiss_index"),
    embeddings,
    allow_dangerous_deserialization=True,
)


@app.get("/search")
@limiter.limit("100/hour")
async def search(
    request: Request,
    q: str,
    min_level: int = 1,
    max_level: int = 5,
    api_key: str = Depends(header_scheme),
):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if min_level > max_level:
        raise HTTPException(
            status_code=400, detail="min_level must be less than max_level"
        )

    results = db.similarity_search(
        q,
        k=10,
        filter={
            "$and": [{"level": {"$gte": min_level}}, {"level": {"$lte": max_level}}]
        },
    )

    list_results = [r.metadata for r in results]

    return {"results": list_results}
