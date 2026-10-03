from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")

# FAQ knowledge base with different questions and answers
faq_data = [
    (
        "What is deep learning?",
        "Deep learning is a branch of machine learning that uses neural networks with multiple layers to learn complex patterns."
    ),
    (
        "How can I install Git?",
        "Download Git from the official Git website and run the installer according to your operating system."
    ),
    (
        "What does a vector embedding mean?",
        "A vector embedding converts information such as text into numerical values that capture its semantic meaning."
    ),
    (
        "Is PyTorch free to use?",
        "Yes, PyTorch is an open-source machine learning framework that can be used freely."
    ),
    (
        "Why is cosine similarity useful?",
        "Cosine similarity compares the direction of two vectors to determine how semantically similar they are."
    ),
]

faq_questions = [question for question, answer in faq_data]
faq_answers = [answer for question, answer in faq_data]

# Convert FAQ questions into embeddings
faq_embeddings = model.encode(faq_questions)


def answer_question(user_query, threshold=0.45):
    query_embedding = model.encode([user_query])

    similarity_scores = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[best_match_index]

    print(f"You asked: {user_query}")

    if best_score < threshold:
        print(
            f"Bot: I couldn't find a relevant answer in my FAQ database. "
            f"(best score: {best_score:.3f})"
        )
    else:
        print(f"Bot: {faq_answers[best_match_index]}")
        print(
            f"(matched question: '{faq_questions[best_match_index]}', "
            f"score: {best_score:.3f})"
        )

    print()


# Test the semantic FAQ bot
answer_question("How do I get Git installed on my computer?")
answer_question("Can you explain neural networks with many layers?")
answer_question("Can I use PyTorch without paying?")
answer_question("What is the temperature outside today?")