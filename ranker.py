from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(resume_texts, job_description):
    # Combine job description and resumes into one list
    docs = [job_description] + resume_texts

    # TF-IDF vectorizer with unigrams and bigrams
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(docs)

    # Vector for job description
    job_vector = tfidf_matrix[0]

    # Vectors for resumes
    resume_vectors = tfidf_matrix[1:]

    # Cosine similarity between job description and each resume
    scores = cosine_similarity(job_vector, resume_vectors)[0]

    # Convert scores to percentage format and round
    ranked = sorted(
        [(i, f"{score * 100:.2f}%") for i, score in enumerate(scores)],
        key=lambda x: float(x[1].strip('%')),
        reverse=True
    )

    return ranked
