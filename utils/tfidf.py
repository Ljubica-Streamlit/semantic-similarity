from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _build_matrix(texts):
    """Fit TF-IDF safely for any number of documents.

    min_df=2 / max_df=0.8 raise errors on small files (e.g. 2-4 pages) or when
    no words survive filtering, so relax them for small inputs.
    """
    n = len(texts)
    if n >= 10:
        params = dict(min_df=2, max_df=0.8)
    else:
        params = dict(min_df=1, max_df=1.0)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        **params
    )
    try:
        return vectorizer.fit_transform(texts)
    except ValueError:
        # e.g. "empty vocabulary" – fall back to the most permissive settings
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=1.0)
        return vectorizer.fit_transform(texts)


def calculate_tfidf_for_pairs(df, url_pairs, progress_callback=None):
    """TF-IDF similarity for specific URL pairs only."""
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].astype(str).tolist()
    url_to_idx = {}
    for idx, url in enumerate(urls):
        url_to_idx.setdefault(url, idx)  # first occurrence wins for duplicate URLs

    tfidf_matrix = _build_matrix(texts)

    results = {}
    total_pairs = len(url_pairs)
    for idx, (url1, url2) in enumerate(url_pairs):
        if progress_callback and idx % 100 == 0:
            progress_callback(idx, max(total_pairs, 1))
        i1, i2 = url_to_idx.get(url1), url_to_idx.get(url2)
        if i1 is not None and i2 is not None:
            score = cosine_similarity(tfidf_matrix[i1], tfidf_matrix[i2])[0][0]
            results[tuple(sorted([url1, url2]))] = float(score)

    if progress_callback:
        progress_callback(total_pairs, max(total_pairs, 1))
    return results


def calculate_tfidf_similarity(df, threshold=0.5, progress_callback=None):
    """TF-IDF similarity for all pairs above threshold."""
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].astype(str).tolist()
    tfidf_matrix = _build_matrix(texts)

    results = []
    n = len(texts)
    block = 500
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
        for local_i in range(end - start):
            i = start + local_i
            for j in range(i + 1, n):
                score = sims[local_i, j]
                if score >= threshold:
                    results.append({'URL 1': urls[i], 'URL 2': urls[j], 'TF-IDF Similarity': float(score)})
        if progress_callback:
            progress_callback(end, n)
    return results
