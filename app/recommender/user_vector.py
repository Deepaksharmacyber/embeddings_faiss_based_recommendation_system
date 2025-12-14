import numpy as np
from .config import ACTIVITY_WEIGHTS

def build_user_vector(user_activity, embedding_index):
    dim = embedding_index.embeddings.shape[1]
    user_vector = np.zeros(dim)

    used = 0

    print("\n🧠 User intent sources:")

    for act in user_activity["activities"]:
        if act["activity_type"] not in ("enrolled", "progress_update"):
            continue

        course_id = act["course_id"]
        if course_id not in embedding_index.course_id_to_index:
            continue

        idx = embedding_index.course_id_to_index[course_id]

        print(f"Using course {course_id} ({act['activity_type']})")

        user_vector += embedding_index.embeddings[idx]
        used += 1

    if used == 0:
        return None

    user_vector /= used
    return user_vector / np.linalg.norm(user_vector)
