import json
import numpy as np
from datetime import datetime

EVENT_WEIGHTS = {
    "view": 0.2,
    "enrolled": 1.0,
    "progress_update": 0.5,
    "completed": 1.5
}

def main():
    course_embeddings = np.load("course_embeddings.npy")

    courses = json.load(open("data/course_sample.json"))
    course_index = {c["course_id"]: i for i, c in enumerate(courses)}

    user_data = json.load(open("data/user_activity_sample.json"))

    user_vector = np.zeros(course_embeddings.shape[1])
    total_weight = 0.0

    for activity in user_data["activities"]:
        course_id = activity["course_id"]
        activity_type = activity["activity_type"]
        progress = activity["progress"]

        base_weight = EVENT_WEIGHTS.get(activity_type, 0)

        # progress-based boost
        progress_weight = progress / 100

        weight = base_weight + progress_weight

        idx = course_index[course_id]
        user_vector += weight * course_embeddings[idx]
        total_weight += weight

    if total_weight > 0:
        user_vector /= total_weight

    np.save("user_embedding.npy", user_vector)
    print("✅ User embedding generated")

if __name__ == "__main__":
    main()
