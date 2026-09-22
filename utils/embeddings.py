import time
import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-small"
MAX_CHARS = 8000
BATCH_SIZE = 50
MAX_RETRIES = 3


def estimate_cost(df, rows_to_embed=None):
    """Rough OpenAI cost estimate (text-embedding-3-small: $0.02 per 1M tokens)."""
    n = len(df) if rows_to_embed is None else rows_to_embed
    avg_tokens = 1000
    return (n * avg_tokens / 1_000_000) * 0.02


def _prepare_text(content):
    text = str(content)[:MAX_CHARS].strip()
    return f"page content: {text}" if text else "page content: (empty)"


def _is_fatal(error_str):
    """Errors that retrying will not fix."""
    s = error_str.lower()
    return any(k in s for k in ('insufficient_quota', 'invalid_api_key', 'incorrect api key', 'account_deactivated'))


def generate_embeddings(df, api_key, progress_callback=None, existing=None):
    """Generate embeddings for every row that does not already have one.

    Always returns a list aligned with df rows (np.ndarray or None), even when
    it stops early, so embeddings that were paid for are never lost.

    Returns dict:
        embeddings   – list aligned with df (None where missing)
        skipped_rows – rows that failed after retries
        error        – fatal error message or None
        new_count    – how many embeddings were generated in this run
    """
    n = len(df)
    embeddings = list(existing) if existing is not None else [None] * n
    todo = [i for i in range(n) if not (isinstance(embeddings[i], np.ndarray) and embeddings[i].size > 0)]
    urls = df['URL'].astype(str).tolist()
    texts = [_prepare_text(c) for c in df['Content'].tolist()]
    skipped_rows = []
    new_count = 0

    def report(done):
        if progress_callback:
            total = max(len(todo), 1)
            progress_callback(done, len(todo), min(done / total, 1.0))

    try:
        client = OpenAI(api_key=api_key, max_retries=2, timeout=60)
    except Exception as e:
        return {'embeddings': embeddings, 'skipped_rows': [], 'error': f"Could not create OpenAI client: {e}", 'new_count': 0}

    def call_api(batch_texts):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=MODEL, input=batch_texts)
                return [np.array(d.embedding, dtype=np.float32) for d in sorted(resp.data, key=lambda d: d.index)], None
            except Exception as e:
                last_error = e
                if _is_fatal(str(e)):
                    raise
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None, last_error

    done = 0
    try:
        for start in range(0, len(todo), BATCH_SIZE):
            batch_idx = todo[start:start + BATCH_SIZE]
            vectors, err = call_api([texts[i] for i in batch_idx])

            if vectors is not None:
                for i, v in zip(batch_idx, vectors):
                    embeddings[i] = v
                    new_count += 1
            else:
                # Batch failed – try rows one by one so one bad page doesn't sink the rest
                for i in batch_idx:
                    single, single_err = call_api([texts[i]])
                    if single is not None:
                        embeddings[i] = single[0]
                        new_count += 1
                    else:
                        skipped_rows.append({'row': i + 1, 'url': urls[i], 'error': str(single_err)})

            done += len(batch_idx)
            report(done)
    except Exception as e:
        msg = str(e)
        if 'insufficient_quota' in msg:
            msg = "OpenAI quota exceeded – please check your billing at platform.openai.com."
        elif 'api_key' in msg.lower() or 'incorrect api key' in msg.lower():
            msg = "OpenAI rejected the API key. Please check it and try again."
        return {'embeddings': embeddings, 'skipped_rows': skipped_rows,
                'error': f"{msg} (stopped after {done}/{len(todo)} pages)", 'new_count': new_count}

    return {'embeddings': embeddings, 'skipped_rows': skipped_rows, 'error': None, 'new_count': new_count}
