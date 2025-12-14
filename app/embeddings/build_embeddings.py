import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    model = SentenceTransformer(MODEL_NAME)

    with open("data/course_sample.json") as f:
        courses = json.load(f)

    texts = []
    for c in courses:
        text = f"""
        Title: {c['title']}
        Description: {c['description']}
        Price: {c['price']}
        Rating: {c['ai_rating_avg']}
        """
        texts.append(text.strip())

    embeddings = model.encode(texts, show_progress_bar=True)

    np.save("course_embeddings.npy", embeddings)
    print("✅ Course embeddings created:", embeddings.shape)

if __name__ == "__main__":
    main()
