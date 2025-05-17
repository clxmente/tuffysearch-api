import os
import json
import faiss

from sentence_transformers import SentenceTransformer

# if the courses don't exist as an array, transform them
if not os.path.exists(os.path.join("data", "courses.json")):
    print("transforming catalog to list of courses...")
    with open(os.path.join("data", "cleaned_2025-2026_catalog.json"), "r") as f:
        catalog = json.load(f)
    with open(os.path.join("data", "courses.json"), "w") as f:
        data = list(catalog.values())
        json.dump(data, f, indent=2)
        print("completed transformation")
else:
    print("courses already exist, skipping transformation...")
    with open(os.path.join("data", "courses.json"), "r") as f:
        data = json.load(f)

# each course will be the department + title + description
courses = []
for course in data:
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
