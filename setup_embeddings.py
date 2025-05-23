import os
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("loading model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("model loaded")

COURSES_FILE = "processed_2025-2026_catalog.json"

with open(os.path.join("data", COURSES_FILE), "r") as f:
    data = json.load(f)

if not os.path.exists(os.path.join("data", "faiss_index")):
    print("faiss index does not exist, creating documents with metadata...")
    # each course will be the department + title + description
    documents = []
    for course in data:
        documents.append(
            Document(
                page_content=f"{course['department']} . {course['title']} . {course['description']}",
                metadata=course,
            )
        )
    db = FAISS.from_documents(documents, embeddings)
    print("documents created, saving faiss index to disk...")
    db.save_local(os.path.join("data", "faiss_index"))
    print("faiss index saved")
else:
    print("faiss index exists, loading from disk...")
    db = FAISS.load_local(
        os.path.join("data", "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("faiss index loaded")

print("searching for courses...")
res = db.similarity_search(
    "income inequality", k=5, filter={"course_level": {"$lte": 399}}
)
print(res)
