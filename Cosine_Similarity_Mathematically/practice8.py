# Predict, then verify

import math


def calculate_dot_product(vector_x, vector_y):
    return sum(x * y for x, y in zip(vector_x, vector_y))


def calculate_magnitude(vector):
    return math.sqrt(sum(value ** 2 for value in vector))


def calculate_cosine_similarity(vector_x, vector_y):
    dot = calculate_dot_product(vector_x, vector_y)
    magnitude_x = calculate_magnitude(vector_x)
    magnitude_y = calculate_magnitude(vector_y)

    return dot / (magnitude_x * magnitude_y)


# Prediction: Both vectors have the same direction,
# so their cosine similarity should be close to 1.0
vector_x = [2, 2, 2]
vector_y = [8, 8, 8]

similarity = calculate_cosine_similarity(vector_x, vector_y)

print("Cosine Similarity:", round(similarity, 4))
print("Matches prediction of 1.0?", round(similarity, 4) == 1.0)