from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def calculate_tfidf_for_pairs(df, url_pairs, progress_callback=None):
    """Calculate TF-IDF similarity for specific URL pairs only"""
    # Prepare texts
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].tolist()
    
    # Create URL to index mapping
    url_to_idx = {url: idx for idx, url in enumerate(urls)}
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarities only for specified pairs
    results = {}
    total_pairs = len(url_pairs)
    
    for idx, (url1, url2) in enumerate(url_pairs):
        if progress_callback and idx % 100 == 0:
            progress_callback(idx, total_pairs)
        
        # Get indices
        idx1 = url_to_idx.get(url1)
        idx2 = url_to_idx.get(url2)
        
        if idx1 is not None and idx2 is not None:
            # Calculate cosine similarity between these two vectors
            vec1 = tfidf_matrix[idx1]
            vec2 = tfidf_matrix[idx2]
            score = cosine_similarity(vec1, vec2)[0][0]
            
            # Store with sorted tuple as key
            key = tuple(sorted([url1, url2]))
            results[key] = score
    
    if progress_callback:
        progress_callback(total_pairs, total_pairs)
    
    return results

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
