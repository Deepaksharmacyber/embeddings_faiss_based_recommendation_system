def recommend_courses(user_vector, embedding_index, user_activity):
    scores, indices = embedding_index.index.search(
        user_vector.reshape(1, -1),
        10
    )

    seen_courses = {a["course_id"] for a in user_activity["activities"]}

    print("\n🎯 Simple similarity-based recommendations:")

    recommendations = []

    for sim_score, idx in zip(scores[0], indices[0]):
        course_id = embedding_index.index_to_course_id[idx]

        if course_id in seen_courses:
            continue

        print(f"Course {course_id} | similarity={sim_score:.3f}")
        recommendations.append(course_id)

    return recommendations[:5]
