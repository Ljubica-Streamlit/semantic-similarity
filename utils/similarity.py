import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    vec_a = np.array(vec_a).reshape(1, -1)
    vec_b = np.array(vec_b).reshape(1, -1)
    return cosine_similarity(vec_a, vec_b)[0][0]

def categorize_priority(similarity_score):
    """Categorize similarity into priority levels"""
    if similarity_score >= 0.95:
        return '🔴 High Priority'
    elif similarity_score >= 0.90:
        return '🟡 Medium Priority'
    elif similarity_score >= 0.85:
        return '🟢 Low Priority'
    else:
        return 'Below Threshold'

def calculate_similarities(df, threshold=0.85, progress_callback=None):
    """Calculate pairwise semantic similarities between all pages"""
    results = []
    
    # Get valid embeddings
    valid_data = df[df['Embedding'].notna()].reset_index(drop=True)
    
    total_comparisons = (len(valid_data) * (len(valid_data) - 1)) // 2
    current_comparison = 0
    
    for i in range(len(valid_data)):
        for j in range(i + 1, len(valid_data)):
            emb_i = valid_data.iloc[i]['Embedding']
            emb_j = valid_data.iloc[j]['Embedding']
            
            # Calculate semantic similarity
            semantic_similarity = calculate_cosine_similarity(emb_i, emb_j)
            
            current_comparison += 1
            if progress_callback and current_comparison % 100 == 0:
                progress_callback(current_comparison, total_comparisons)
            
            # Only include if above threshold
            if semantic_similarity >= threshold:
                url1 = valid_data.iloc[i]['URL']
                url2 = valid_data.iloc[j]['URL']
                
                result = {
                    'URL 1': url1,
                    'URL 2': url2,
                    'Semantic Similarity': semantic_similarity
                }
                
                # Add categories if available
                if 'Category' in valid_data.columns:
                    cat1 = valid_data.iloc[i]['Category'] if pd.notna(valid_data.iloc[i]['Category']) else 'N/A'
                    cat2 = valid_data.iloc[j]['Category'] if pd.notna(valid_data.iloc[j]['Category']) else 'N/A'
                    result['Category 1'] = cat1
                    result['Category 2'] = cat2
                    result['Same Category?'] = cat1 == cat2
                
                results.append(result)
    
    if progress_callback:
        progress_callback(total_comparisons, total_comparisons)
    
    return results

import pandas as pd
