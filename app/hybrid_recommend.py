import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------
# Hybrid weights
# ---------------------------------------
WEIGHTS = {
    "user_similarity": 0.55,
    "content_similarity": 0.25,
    "popularity": 0.20
}

EVENT_WEIGHTS = {
    "view": 0.3,
    "enrolled": 1.0
}

MAX_FAISS_CANDIDATES = 50


# ---------------------------------------
# Helpers
# ---------------------------------------

def normalize(values):
    min_v, max_v = min(values), max(values)
    if max_v - min_v == 0:
        return [0.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


# ---------------------------------------
# Hybrid Recommendation Engine
# ---------------------------------------

def hybrid_recommend(top_k=5):
    print("\n🚀 Starting Hybrid Recommendation Engine")

    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------
    # Load data
    # ---------------------------
    courses = json.load(open("data/course_sample.json"))
    popularity = json.load(open("data/popularity_sample.json"))
    user_activity = json.load(open("data/user_activity_sample.json"))

    course_by_index = {i: c for i, c in enumerate(courses)}
    course_by_id = {c["course_id"]: c for c in courses}

    seen_courses = {a["course_id"] for a in user_activity["activities"]}

    print(f"📦 Total courses: {len(courses)}")
    print(f"👀 Seen courses: {seen_courses}")

    # ---------------------------
    # Build USER EMBEDDING (views + enrollments only)
    # ---------------------------
    course_embeddings = np.load("course_embeddings.npy")
    user_vector = np.zeros(course_embeddings.shape[1])
    total_weight = 0.0

    for a in user_activity["activities"]:
        cid = a["course_id"]
        event = a["activity_type"]

        if event not in EVENT_WEIGHTS:
            continue

        weight = EVENT_WEIGHTS[event]
        user_vector += weight * course_embeddings[cid - 1]
        total_weight += weight

    if total_weight > 0:
        user_vector /= total_weight

    user_vector = user_vector.reshape(1, -1)

    print("🧠 User embedding built from views + enrollments")

    # ---------------------------
    # FAISS user similarity
    # ---------------------------
    index = faiss.read_index("index/faiss.index")
    k = min(MAX_FAISS_CANDIDATES, index.ntotal)

    distances, indices = index.search(user_vector, k)

    candidates = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        course = course_by_index[idx]
        cid = course["course_id"]

        # if cid in seen_courses:
        #     continue

        enrolled_courses = {
            a["course_id"]
            for a in user_activity["activities"]
            if a["activity_type"] == "enrolled"
        }

        if cid in enrolled_courses:
            print(f'enrolled courses {cid}')
            continue

        candidates.append({
            "course": course,
            "user_similarity": 1 / (1 + dist),
            "content_similarity": 0.0,
            "popularity": 0.0
        })

    print(f"🔍 Candidates after FAISS & seen-filter: {len(candidates)}")

    if not candidates:
        print("❌ No candidates found")
        return

    # ---------------------------
    # Content similarity (last enrolled / viewed)
    # ---------------------------
    last_event = user_activity["activities"][-1]
    last_course = course_by_id[last_event["course_id"]]

    content_text = f"{last_course['title']} {last_course['description']}"
    content_vector = model.encode([content_text])

    content_distances, content_indices = index.search(content_vector, k)

    content_score_map = {}
    for idx, dist in zip(content_indices[0], content_distances[0]):
        if idx == -1:
            continue
        cid = course_by_index[idx]["course_id"]
        content_score_map[cid] = 1 / (1 + dist)

    for c in candidates:
        cid = c["course"]["course_id"]
        c["content_similarity"] = content_score_map.get(cid, 0.0)

    print("📐 Content similarity computed")

    # ---------------------------
    # Popularity
    # ---------------------------
    for c in candidates:
        cid = str(c["course"]["course_id"])
        pop = popularity.get(cid, {"enrollments": 0, "avg_progress": 0})
        c["popularity"] = pop["enrollments"]

    print("🔥 Popularity scores added")

    # ---------------------------
    # Normalize scores
    # ---------------------------
    for key in ["user_similarity", "content_similarity", "popularity"]:
        vals = [c[key] for c in candidates]
        norm_vals = normalize(vals)
        for i, c in enumerate(candidates):
            c[key] = norm_vals[i]

    # ---------------------------
    # Final hybrid score
    # ---------------------------
    for c in candidates:
        c["final_score"] = (
            WEIGHTS["user_similarity"] * c["user_similarity"] +
            WEIGHTS["content_similarity"] * c["content_similarity"] +
            WEIGHTS["popularity"] * c["popularity"]
        )

    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # ---------------------------
    # Output
    # ---------------------------
    print("\n📌 Final Hybrid Recommendations:")
    for r in candidates[:top_k]:
        print(
            f"- {r['course']['title']} "
            f"(score={round(r['final_score'], 3)})"
        )


if __name__ == "__main__":
    hybrid_recommend()
