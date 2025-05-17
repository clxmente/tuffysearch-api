import os
import json
import faiss

from sentence_transformers import SentenceTransformer

# each course will be the department + title + description
courses = []

with open(os.path.join("data", "courses.json"), "r") as f:
    data = json.load(f)

for course in data.values():
    courses.append(f"{course['department']} {course['title']} {course['description']}")

# load the model and generate the embeddings for our courses
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(courses)
embeddings = embeddings.astype("float32")

# create the faiss index to store so we don't have to recompute each startup
index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)  # type: ignore

faiss.write_index(index, os.path.join("data", "course_index.faiss"))
