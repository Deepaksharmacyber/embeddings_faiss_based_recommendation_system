import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def recommend(course_id, top_k=3):
    model = SentenceTransformer(MODEL_NAME)

    with open("data/courses.json") as f:
        courses = json.load(f)

    course_map = {c["id"]: c for c in courses}
    target = course_map[course_id]

    query_text = f"""
    Category: {target['category']}
    Domain: {target['domain']}
    Language: {target['language']}
    Level: {target['level']}
    Title: {target['title']}
    Description: {target['description']}
    """

    query_vector = model.encode([query_text])

    print("\n🔎 Query Text Used:")
    print(query_text)

    index = faiss.read_index("index/faiss.index")

    distances, indices = index.search(query_vector, top_k + 1)



    print("\n📌 Recommendations:")
    for idx in indices[0]:
        course = courses[idx]
        if course["id"] != course_id:
            print("-", course["title"])

def recommend_for_user(top_k=3):
    model = SentenceTransformer(MODEL_NAME)

    with open("data/courses.json") as f:
        courses = json.load(f)

    user_vector = np.load("user_embedding.npy").reshape(1, -1)

    index = faiss.read_index("index/faiss.index")

    distances, indices = index.search(user_vector, top_k + 3)

    print("\n📌 User-based Recommendations:")
    seen_courses = {event["course_id"] for event in json.load(open("data/user_events.json"))["events"]}

    count = 0
    for idx in indices[0]:
        course = courses[idx]

        if course["id"] in seen_courses:
            continue

        print("-", course["title"])
        count += 1

        if count == top_k:
            break

if __name__ == "__main__":
    # recommend(course_id=
    recommend_for_user()
