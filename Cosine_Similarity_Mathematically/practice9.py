# Real sentence embeddings instead of toy vectors

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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


model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_x = "I enjoy exploring artificial intelligence and neural networks."
sentence_y = "I am interested in machine learning and deep learning."

embedding_x = model.encode(sentence_x)
embedding_y = model.encode(sentence_y)

# Calculate similarity manually using the generated embeddings
manual_similarity = calculate_cosine_similarity(
    embedding_x,
    embedding_y
)

# Calculate similarity using scikit-learn
sklearn_similarity = cosine_similarity(
    [embedding_x],
    [embedding_y]
)[0][0]

print("Manual cosine similarity:", round(manual_similarity, 4))
print("Scikit-learn result:     ", round(sklearn_similarity, 4))
print(
    "Results match?",
    round(manual_similarity, 4) == round(sklearn_similarity, 4)
)