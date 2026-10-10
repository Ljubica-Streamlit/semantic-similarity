import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_matrix(texts, filter_common=True):
    """Fit TF-IDF safely for any number of documents. Rows are L2-normalised.

    min_df=2 / max_df=0.8 raise errors on small files (e.g. 2-4 pages) or when
    no words survive filtering, so relax them for small inputs.
    """
    n = len(texts)
    if not filter_common:
        params = dict(min_df=1, max_df=1.0)
    elif n >= 10:
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


PAIR_CHUNK = 50_000


def calculate_tfidf_for_pairs(df, url_pairs, progress_callback=None, filter_common=True, tfidf_matrix=None):
    """TF-IDF similarity for specific URL pairs only.

    Pass `tfidf_matrix` (from build_matrix) to reuse a matrix already fitted on df['Content'].
    """
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].astype(str).tolist()
    url_to_idx = {}
    for idx, url in enumerate(urls):
        url_to_idx.setdefault(url, idx)  # first occurrence wins for duplicate URLs

    if tfidf_matrix is None:
        tfidf_matrix = build_matrix(texts, filter_common)

    known = [(u1, u2) for u1, u2 in url_pairs if u1 in url_to_idx and u2 in url_to_idx]
    results = {}
    total_pairs = len(url_pairs)
    # Rows are L2-normalised, so cosine similarity is the row-wise dot product
    for start in range(0, len(known), PAIR_CHUNK):
        chunk = known[start:start + PAIR_CHUNK]
        i1 = [url_to_idx[u1] for u1, _ in chunk]
        i2 = [url_to_idx[u2] for _, u2 in chunk]
        scores = np.asarray(tfidf_matrix[i1].multiply(tfidf_matrix[i2]).sum(axis=1)).ravel()
        for (url1, url2), score in zip(chunk, scores):
            results[tuple(sorted([url1, url2]))] = float(score)
        if progress_callback:
            progress_callback(min(start + PAIR_CHUNK, total_pairs), max(total_pairs, 1))

    if progress_callback:
        progress_callback(total_pairs, max(total_pairs, 1))
    return results


def calculate_tfidf_similarity(df, threshold=0.5, progress_callback=None, filter_common=True, tfidf_matrix=None):
    """TF-IDF similarity for all pairs above threshold."""
    texts = df['Content'].fillna('').astype(str).tolist()
    urls = df['URL'].astype(str).tolist()
    if tfidf_matrix is None:
        tfidf_matrix = build_matrix(texts, filter_common)

    results = []
    n = len(texts)
    block = 500
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i, i + 1:]
            for h in np.nonzero(row >= threshold)[0]:
                results.append({'URL 1': urls[i], 'URL 2': urls[i + 1 + h], 'TF-IDF Similarity': float(row[h])})
        if progress_callback:
            progress_callback(end, n)
    return results
