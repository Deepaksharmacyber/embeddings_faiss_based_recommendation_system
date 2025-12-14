import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------
# Category taxonomy
# --------------------------------------------------

CATEGORY_KEYWORDS = {
    "backend": ["python", "django", "backend", "api", "fastapi", "server", "microservice"],
    "frontend": ["react", "javascript", "frontend", "html", "css"],
    "devops": ["docker", "kubernetes", "deployment", "ci/cd"],
    "database": ["sql", "postgres", "database"],
    "system_design": ["system design", "scalable", "architecture"],
    "data": ["data", "pandas", "numpy", "analysis"],
    "ml": ["machine learning", "ml", "model"]
}

# Hybrid weights (category handled by gating, not scoring)
WEIGHTS = {
    "user_similarity": 0.45,
    "content_similarity": 0.35,
    "popularity": 0.20
}

MAX_FAISS_CANDIDATES = 30


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize(values):
    min_v, max_v = min(values), max(values)
    if max_v - min_v == 0:
        return [0.0] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


def infer_user_interests(user_activity, course_by_id):
    """
    Learn multi-category interest distribution from user behavior.
    """
    interests = {k: 0.0 for k in CATEGORY_KEYWORDS.keys()}

    for a in user_activity["activities"]:
        course = course_by_id[a["course_id"]]
        text = (course["title"] + " " + course["description"]).lower()

        weight = 1.0
        if a["activity_type"] == "enrolled":
            weight = 2.0
        elif a["activity_type"] == "progress_update":
            weight = 2.5

        weight += a["progress"] / 100

        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(k in text for k in keywords):
                interests[category] += weight

    total = sum(interests.values()) or 1.0
    for k in interests:
        interests[k] /= total

    return interests


def get_primary_category(user_interests):
    """
    Dominant user intent (single strongest category).
    """
    return max(user_interests.items(), key=lambda x: x[1])[0]


def course_matches_primary_category(course, primary_category):
    """
    Gate courses by primary user intent.
    """
    text = (course["title"] + " " + course["description"]).lower()
    return any(k in text for k in CATEGORY_KEYWORDS[primary_category])


# --------------------------------------------------
# Hybrid Recommendation Engine
# --------------------------------------------------

def hybrid_recommend(top_k=5):
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

    # ---------------------------
    # Infer user intent
    # ---------------------------
    user_interests = infer_user_interests(user_activity, course_by_id)
    primary_category = get_primary_category(user_interests)

    print("\n🧠 User interest distribution:")
    for k, v in sorted(user_interests.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k}: {round(v, 2)}")

    print(f"\n🎯 Primary user intent: {primary_category}")

    # ---------------------------
    # Load vectors
    # ---------------------------
    user_vector = np.load("user_embedding.npy").reshape(1, -1)
    index = faiss.read_index("index/faiss.index")

    k = min(MAX_FAISS_CANDIDATES, index.ntotal)

    # ---------------------------
    # 1️⃣ FAISS user similarity
    # ---------------------------
    user_distances, user_indices = index.search(user_vector, k)

    candidates = []

    for idx, dist in zip(user_indices[0], user_distances[0]):
        if idx == -1:
            continue

        course = course_by_index[idx]

        # 🔥 PRIMARY-INTENT GATE (CORE FIX)
        if not course_matches_primary_category(course, primary_category):
            continue

        candidates.append({
            "course": course,
            "user_similarity": 1 / (1 + dist),
            "content_similarity": 0.0,
            "popularity": 0.0
        })

    if not candidates:
        print("\n⚠️ No candidates after intent gating.")
        return

    # ---------------------------
    # 2️⃣ Content similarity (last strong intent)
    # ---------------------------
    last_strong = next(
        a for a in reversed(user_activity["activities"])
        if a["activity_type"] in ("enrolled", "progress_update")
    )

    last_course = course_by_id[last_strong["course_id"]]

    content_text = f"""
    Title: {last_course['title']}
    Description: {last_course['description']}
    """

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

    # ---------------------------
    # 3️⃣ Popularity
    # ---------------------------
    for c in candidates:
        cid = str(c["course"]["course_id"])
        pop = popularity.get(cid, {"enrollments": 0, "avg_progress": 0})
        c["popularity"] = pop["enrollments"] * 0.7 + pop["avg_progress"] * 0.3

    # ---------------------------
    # 4️⃣ Normalize scores
    # ---------------------------
    for key in ["user_similarity", "content_similarity", "popularity"]:
        vals = [c[key] for c in candidates]
        norm_vals = normalize(vals)
        for i, c in enumerate(candidates):
            c[key] = norm_vals[i]

    # ---------------------------
    # 5️⃣ Final hybrid score
    # ---------------------------
    final_list = []

    for c in candidates:
        cid = c["course"]["course_id"]
        if cid in seen_courses:
            continue

        c["final_score"] = (
            WEIGHTS["user_similarity"] * c["user_similarity"] +
            WEIGHTS["content_similarity"] * c["content_similarity"] +
            WEIGHTS["popularity"] * c["popularity"]
        )

        final_list.append(c)

    final_list.sort(key=lambda x: x["final_score"], reverse=True)

    # ---------------------------
    # Output
    # ---------------------------
    print("\n📌 Final Hybrid Recommendations:")
    for r in final_list[:top_k]:
        print("-", r["course"]["title"])


if __name__ == "__main__":
    hybrid_recommend()
