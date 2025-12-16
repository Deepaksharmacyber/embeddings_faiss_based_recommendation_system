import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("🚀 Building course embeddings")

    model = SentenceTransformer(MODEL_NAME)

    courses = json.load(open("data/course_sample.json"))

    texts = [
        f"{c['title']} {c['description']}"
        for c in courses
    ]

    embeddings = model.encode(texts, show_progress_bar=True)

    np.save("course_embeddings.npy", embeddings)

    print(f"✅ Saved course_embeddings.npy ({len(embeddings)} courses)")


if __name__ == "__main__":
    main()
