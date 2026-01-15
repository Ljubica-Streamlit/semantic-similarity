from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_tfidf_similarity(df, threshold=0.5, progress_callback=None):
    """Calculate TF-IDF based similarity"""
    # Prepare texts
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].tolist()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    results = []
    total_comparisons = (len(texts) * (len(texts) - 1)) // 2
    current_comparison = 0
    
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = similarity_matrix[i][j]
            
            current_comparison += 1
            if progress_callback and current_comparison % 1000 == 0:
                progress_callback(current_comparison, total_comparisons)
            
            if score >= threshold:
                results.append({
                    'URL 1': urls[i],
                    'URL 2': urls[j],
                    'TF-IDF Similarity': score
                })
    
    if progress_callback:
        progress_callback(total_comparisons, total_comparisons)
    
    return results
