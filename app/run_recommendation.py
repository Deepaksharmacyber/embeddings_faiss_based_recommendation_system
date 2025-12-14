import json
from recommender.embedding_index import CourseEmbeddingIndex
from recommender.user_vector import build_user_vector
from recommender.recommend import recommend_courses


COURSE_PATH = "data/course_sample.json"
POPULARITY_PATH = "data/popularity_sample.json"
USER_ACTIVITY_PATH = "data/user_activity_sample.json"


def main():
    with open(USER_ACTIVITY_PATH) as f:
        user_activity = json.load(f)

    embedding_index = CourseEmbeddingIndex(COURSE_PATH)

    user_vector = build_user_vector(user_activity, embedding_index)

    recommendations = recommend_courses(
        user_vector,
        embedding_index,
        user_activity
    )

    print("\n🎯 Final Recommended Course IDs:")
    for cid in recommendations[:5]:
        print("-", cid)


if __name__ == "__main__":
    main()
