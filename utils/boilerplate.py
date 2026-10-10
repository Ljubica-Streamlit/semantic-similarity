"""Remove text blocks that repeat across most pages (nav, footer, cookie banners, CTAs).

Only what is sent for embedding changes – the original content is never modified.
"""
import re
from collections import Counter

MIN_PAGES = 10          # below this there is not enough repetition to judge
MIN_BLOCK_CHARS = 8     # ignore single words
DEFAULT_THRESHOLD = 0.8  # a block must appear on this share of pages to be removed
THIN_PAGE_MAX_UNIQUE_PCT = 40  # below this a page is mostly shared template

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?:])\s+|(?<=[.!?])(?=[A-Z])|\n+|\s{3,}|(?<=[a-z])(?=[A-Z][a-z]+\s)')
_WS = re.compile(r'\s+')


def split_blocks(text):
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(str(text)) if p and p.strip()]
    return parts


def _key(block):
    return _WS.sub(' ', block.strip().lower())


def analyse(texts, threshold=DEFAULT_THRESHOLD, min_pages=MIN_PAGES):
    """Find blocks that appear on >= threshold share of pages.

    Returns (cleaned_texts, stats). When there are too few pages, the texts come
    back unchanged and stats['applied'] is False.
    """
    n = len(texts)
    blocks_per_page = [split_blocks(t) for t in texts]
    stats = {
        'applied': False, 'pages': n, 'threshold': threshold,
        'removed_blocks': [], 'removed_chars_avg': 0, 'unique_pct': [100.0] * n,
        'heavy': False, 'reason': '',
    }
    if n < min_pages:
        stats['reason'] = f"Fewer than {min_pages} pages – not enough repetition to judge."
        return list(texts), stats

    counts = Counter()
    for blocks in blocks_per_page:
        for key in {_key(b) for b in blocks if len(b) >= MIN_BLOCK_CHARS}:
            counts[key] += 1

    cutoff = max(2, int(round(threshold * n)))
    common = {k for k, c in counts.items() if c >= cutoff}
    if not common:
        stats['reason'] = "No blocks repeat on enough pages."
        return list(texts), stats

    cleaned, removed_chars, unique_pct = [], [], []
    for blocks in blocks_per_page:
        kept = [b for b in blocks if _key(b) not in common]
        original_len = sum(len(b) for b in blocks) or 1
        new_text = ' '.join(kept)
        cleaned.append(new_text)
        removed_chars.append(original_len - len(new_text))
        unique_pct.append(round(100 * min(len(new_text) / original_len, 1.0), 1))

    examples = sorted(((counts[k], k) for k in common), reverse=True)[:25]
    stats.update(
        applied=True,
        removed_blocks=[{'pages': c, 'share': round(100 * c / n), 'text': (t[:160] + '…') if len(t) > 160 else t}
                        for c, t in examples],
        removed_block_count=len(common),
        removed_chars_avg=int(sum(removed_chars) / n),
        unique_pct=unique_pct,
        avg_unique_pct=round(sum(unique_pct) / n, 1),
    )
    stats['heavy'] = stats['avg_unique_pct'] < 30
    stats['empty_pages'] = int(sum(1 for t in cleaned if len(t.strip()) < 30))
    return cleaned, stats


def is_thin(unique_pct):
    return unique_pct < THIN_PAGE_MAX_UNIQUE_PCT


def thin_page_note(unique_pct, cleaned_text):
    """Short label for pages that are mostly shared template."""
    if is_thin(unique_pct):
        return f"⚠️ mostly shared template ({len(cleaned_text.strip())} unique characters)"
    return ''
