import time
import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-small"
MODEL_TOKEN_LIMIT = 8191          # text-embedding-3 models
MAX_TOKENS = 8000                 # leave a little headroom
CHARS_PER_TOKEN = 3.5             # conservative fallback when tiktoken is unavailable
BATCH_SIZE = 50
MAX_RETRIES = 3

try:  # exact token counting when available
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - optional dependency
    _ENC = None


def trim_to_limit(text, max_tokens=MAX_TOKENS):
    """Cut text to the model's input limit. Returns (text, was_trimmed)."""
    text = str(text)
    if _ENC is not None:
        tokens = _ENC.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text, False
        return _ENC.decode(tokens[:max_tokens]), True
    limit = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def estimate_cost(texts, model=MODEL):
    """Rough OpenAI cost estimate from the actual text length."""
    if _ENC is not None:
        tokens = sum(min(len(_ENC.encode(str(t), disallowed_special=())), MAX_TOKENS) for t in texts)
    else:
        tokens = sum(min(len(str(t)) // CHARS_PER_TOKEN, MAX_TOKENS) for t in texts)
    price = 0.13 if 'large' in model else 0.02  # per 1M tokens
    return tokens / 1_000_000 * price


def _is_fatal(error_str):
    s = error_str.lower()
    return any(k in s for k in ('insufficient_quota', 'invalid_api_key', 'incorrect api key', 'account_deactivated'))


def generate_embeddings(texts, urls, api_key, progress_callback=None, existing=None, model=MODEL):
    """Embed every text that does not already have an embedding.

    Always returns a list aligned with `texts` (np.ndarray or None), even when it
    stops early, so embeddings that were paid for are never lost.
    """
    n = len(texts)
    embeddings = list(existing) if existing is not None else [None] * n
    todo = [i for i in range(n) if not (isinstance(embeddings[i], np.ndarray) and embeddings[i].size > 0)]
    skipped_rows = []
    new_count = 0
    trimmed = 0

    prepared = []
    for t in texts:
        text = str(t).strip() or "(empty page)"
        text, was_trimmed = trim_to_limit(text)
        trimmed += int(was_trimmed)
        prepared.append(f"page content: {text}")

    def report(done):
        if progress_callback:
            total = max(len(todo), 1)
            progress_callback(done, len(todo), min(done / total, 1.0))

    try:
        client = OpenAI(api_key=api_key, max_retries=2, timeout=90)
    except Exception as e:
        return {'embeddings': embeddings, 'skipped_rows': [], 'error': f"Could not create OpenAI client: {e}",
                'new_count': 0, 'trimmed': trimmed, 'model': model}

    def call_api(batch_texts):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(model=model, input=batch_texts)
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
            vectors, err = call_api([prepared[i] for i in batch_idx])

            if vectors is not None:
                for i, v in zip(batch_idx, vectors):
                    embeddings[i] = v
                    new_count += 1
            else:
                for i in batch_idx:
                    single, single_err = call_api([prepared[i]])
                    if single is None and 'maximum context length' in str(single_err).lower():
                        # token estimate was too generous – halve the text and try once more
                        single, single_err = call_api([prepared[i][:len(prepared[i]) // 2]])
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
                'error': f"{msg} (stopped after {done}/{len(todo)} pages)", 'new_count': new_count,
                'trimmed': trimmed, 'model': model}

    return {'embeddings': embeddings, 'skipped_rows': skipped_rows, 'error': None,
            'new_count': new_count, 'trimmed': trimmed, 'model': model}
