import json
import numpy as np

EVENT_WEIGHTS = {
    "view": 0.3,
    "enrolled": 1.0,
}

def main():
    # Load course embeddings
    course_embeddings = np.load("course_embeddings.npy")

    # Load course data
    courses = json.load(open("data/course_sample.json"))
    course_index = {c["course_id"]: i for i, c in enumerate(courses)}

    # Load user activity
    user_data = json.load(open("data/user_activity_sample.json"))

    # Initialize user vector
    user_vector = np.zeros(course_embeddings.shape[1])
    total_weight = 0.0

    for activity in user_data["activities"]:
        course_id = activity["course_id"]
        activity_type = activity["activity_type"]

        if activity_type not in EVENT_WEIGHTS:
            continue

        weight = EVENT_WEIGHTS[activity_type]

        idx = course_index[course_id]
        user_vector += weight * course_embeddings[idx]
        total_weight += weight

        print(f"✔ Using activity: {activity_type} on course_id={course_id}, weight={weight}")

    if total_weight > 0:
        user_vector /= total_weight

    np.save("user_embedding.npy", user_vector)
    print("\n✅ User embedding generated and saved as user_embedding.npy")


if __name__ == "__main__":
    main()
