import faiss
import numpy as np


def main():
    print("🚀 Building FAISS index")

    embeddings = np.load("course_embeddings.npy")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "index/faiss.index")

    print(f"✅ FAISS index built with {index.ntotal} courses")


if __name__ == "__main__":
    main()
