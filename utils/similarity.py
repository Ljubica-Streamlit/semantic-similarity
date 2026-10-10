import numpy as np

# Similarity levels, checked from the top down. (min score, label, short description)
SIMILARITY_LEVELS = [
    (0.999, "🟥 Duplicate", "100% – identical content"),
    (0.98, "🔴 Near-duplicate", "98–99.9% – almost the same content"),
    (0.95, "🟠 Very similar", "95–98% – same topic and angle"),
    (0.90, "🟡 Similar", "90–95% – strong topical overlap"),
    (0.85, "🟢 Related", "85–90% – same theme, different focus"),
    (0.0, "⚪ Loosely related", "below 85% – some shared context"),
]

LEVEL_ORDER = [label for _, label, _ in SIMILARITY_LEVELS]


def categorize_similarity(score):
    """Map a cosine similarity score (0-1) to a similarity level label."""
    for min_score, label, _ in SIMILARITY_LEVELS:
        if score >= min_score:
            return label
    return SIMILARITY_LEVELS[-1][1]


def calculate_cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _is_valid(x):
    return isinstance(x, np.ndarray) and x.size > 0


def calculate_similarities(df, threshold=0.85, progress_callback=None, block_size=500):
    """Pairwise cosine similarity for all rows with a valid embedding.

    Vectorised in blocks, so 5,000 pages takes seconds instead of hours.
    Rows whose embedding is missing or has a different length are skipped.
    """
    mask = df['Embedding'].apply(_is_valid)
    valid = df[mask].reset_index(drop=True)
    if len(valid) < 2:
        return []

    # Keep only embeddings with the most common dimension (guards against mixed models)
    dims = valid['Embedding'].apply(len)
    main_dim = dims.mode()[0]
    valid = valid[dims == main_dim].reset_index(drop=True)
    n = len(valid)
    if n < 2:
        return []

    matrix = np.vstack(valid['Embedding'].values).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    urls = valid['URL'].astype(str).tolist()
    has_cat = 'Category' in valid.columns
    cats = valid['Category'].where(valid['Category'].notna(), 'N/A').tolist() if has_cat else None

    results = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = matrix[start:end] @ matrix.T
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i, i + 1:]
            hits = np.nonzero(row >= threshold)[0]
            for h in hits:
                j = i + 1 + h
                score = float(min(row[h], 1.0))
                result = {'URL 1': urls[i], 'URL 2': urls[j], 'Semantic Similarity': score}
                if has_cat:
                    result['Category 1'] = cats[i]
                    result['Category 2'] = cats[j]
                    result['Same Category?'] = cats[i] == cats[j]
                results.append(result)
        if progress_callback:
            progress_callback(end, n)

    results.sort(key=lambda r: r['Semantic Similarity'], reverse=True)
    return results


# --------------------------------------------------------------- overlap types
FULL_OVERLAP_HIGH = 0.70     # raw-text word overlap counted as "same words"
TEMPLATE_TWIN_MIN = 0.80     # raw overlap that makes a pair worth showing on its own
MAX_TEMPLATE_TWIN_PAIRS = 5000


def classify_overlap(unique_sim, full_overlap, threshold):
    """Combine unique-content similarity with full-page overlap into one label."""
    import math
    has_unique = unique_sim is not None and not (isinstance(unique_sim, float) and math.isnan(unique_sim))
    full = 0.0 if full_overlap is None else float(full_overlap)
    if has_unique and unique_sim >= threshold:
        return 'Same content' if full >= FULL_OVERLAP_HIGH else 'Same topic, different wording'
    if full >= TEMPLATE_TWIN_MIN:
        return 'Template twins (shared blocks)'
    return 'Related'
