# Cosine similarity using list comprehensions

import math


def dot_product(vector_x, vector_y):
    return sum(x * y for x, y in zip(vector_x, vector_y))


def vector_magnitude(vector):
    return math.sqrt(sum(value ** 2 for value in vector))


def calculate_cosine_similarity(vector_x, vector_y):
    product = dot_product(vector_x, vector_y)
    magnitude_x = vector_magnitude(vector_x)
    magnitude_y = vector_magnitude(vector_y)

    return product / (magnitude_x * magnitude_y)


vector_x = [3, 1, 4]
vector_y = [2, 5, 3]

similarity = calculate_cosine_similarity(vector_x, vector_y)

print("Cosine Similarity:", round(similarity, 4))