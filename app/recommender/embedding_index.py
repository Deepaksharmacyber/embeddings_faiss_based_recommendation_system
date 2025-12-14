import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_NAME


class CourseEmbeddingIndex:
    def __init__(self, course_json_path):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.courses = self._load_courses(course_json_path)
        self.texts, self.course_id_to_index, self.index_to_course_id = self._prepare_texts()
        self.embeddings = self._create_embeddings()
        self.index = self._build_faiss_index()

    def _load_courses(self, path):
        with open(path) as f:
            courses = json.load(f)
        return [c for c in courses if c["is_published"]]

    def _prepare_texts(self):
        texts = []
        cid_to_idx = {}
        idx_to_cid = {}

        for idx, course in enumerate(self.courses):
            text = f"{course['title']}. {course['description']}"
            texts.append(text)
            cid_to_idx[course["course_id"]] = idx
            idx_to_cid[idx] = course["course_id"]

        return texts, cid_to_idx, idx_to_cid

    def _create_embeddings(self):
        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def _build_faiss_index(self):
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.embeddings)
        return index
