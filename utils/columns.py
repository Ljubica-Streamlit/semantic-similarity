"""Load any CSV/Excel file and work out which column is which.

Headers are optional: columns are detected by what they contain.
"""
import io
import re
import string

import pandas as pd

URL_RE = re.compile(r'^(https?://|www\.)\S+$|^[a-z0-9][\w.-]*\.[a-z]{2,}(/\S*)?$|^/\S*$', re.I)
EMBED_RE = re.compile(r'^\s*\[\s*-?\d')

URL_NAMES = {'url', 'urls', 'address', 'page url', 'page', 'link', 'links', 'landing page', 'permalink', 'loc', 'uri'}
CONTENT_NAMES = {'content', 'contents', 'text', 'body', 'body text', 'page text', 'page content', 'main content',
                 'copy', 'html text', 'article', 'description', 'extracted text', 'content 1', 'extractor 1 1'}
CATEGORY_NAMES = {'category', 'categories', 'type', 'page type', 'section', 'cluster', 'group', 'topic',
                  'template', 'segment', 'folder'}
EMBED_NAMES = {'embedding', 'embeddings', 'ai embedding', 'vector', 'vectors'}

MAX_HEADER_SCAN = 5


def _norm(name):
    return re.sub(r'\s+', ' ', str(name).strip().lower())


def col_letter(i):
    letters = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = string.ascii_uppercase[r] + letters
    return letters


# ----------------------------------------------------------------- loading
def load_table(file_name, raw_bytes):
    """Return a DataFrame of strings with NO header applied (header=None)."""
    name = file_name.lower()
    if name.endswith(('.xlsx', '.xlsm', '.xls')):
        sheets = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=None, header=None, dtype=object)
        # pick the sheet with the most filled cells
        df = max(sheets.values(), key=lambda s: s.notna().sum().sum())
    else:
        df = _read_csv(raw_bytes)
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    df.columns = range(df.shape[1])
    df = df.map(_to_str)
    if name.endswith(('.xlsx', '.xlsm', '.xls')):
        # Excel stores control characters as _x000B_ etc. – turn them into spaces
        df = df.replace(r'_x[0-9A-Fa-f]{4}_', ' ', regex=True)
    return df


def _to_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _read_csv(raw):
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = sum(1 for ln in text.splitlines() if ln.strip())
    best, best_score = None, None
    for sep in (',', ';', '\t', '|'):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str,
                             keep_default_na=False, skip_blank_lines=True, on_bad_lines='skip')
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        # Prefer the separator that keeps the most rows intact, then the fewest columns
        # (a wrong separator splits long text into many ragged pieces)
        filled = (df != '').sum(axis=1)
        consistency = (filled == filled.mode()[0]).mean() if len(df) else 0
        score = (len(df) / max(lines, 1) + consistency, -df.shape[1])
        if best_score is None or score > best_score:
            best, best_score = df, score
    if best is None:
        best = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False,
                           on_bad_lines='skip')
    return best


# ----------------------------------------------------------------- detection
def _profile(series):
    vals = [v for v in series.tolist() if v]
    if not vals:
        return dict(fill=0, url=0, embed=0, avg_len=0, unique=0, numeric=0)
    sample = vals[:1000]
    n = len(sample)
    return dict(
        fill=len(vals) / max(len(series), 1),
        url=sum(bool(URL_RE.match(v)) for v in sample) / n,
        embed=sum(bool(EMBED_RE.match(v)) for v in sample) / n,
        avg_len=sum(len(v) for v in sample) / n,
        unique=len(set(sample)) / n,
        numeric=sum(v.replace('.', '', 1).replace('-', '', 1).isdigit() for v in sample) / n,
    )


def detect_columns(raw):
    """Return dict: header_row (int or None), url, content, category, embedding (col index or None)."""
    ncols = raw.shape[1]
    body = raw.iloc[1:] if len(raw) > 1 else raw
    prof = {c: _profile(body[c]) for c in range(ncols)}

    # Candidate URL column by data
    url_col = max(range(ncols), key=lambda c: (prof[c]['url'], prof[c]['fill']), default=None)
    if url_col is not None and prof[url_col]['url'] < 0.5:
        url_col = None

    # Header row = the row right before the first URL-looking cell in the URL column
    header_row = None
    if url_col is not None:
        for r in range(min(MAX_HEADER_SCAN + 1, len(raw))):
            if URL_RE.match(raw.iat[r, url_col] or ''):
                header_row = r - 1 if r > 0 else None
                break
    else:
        # No URL column found: treat the first row as header if its cells are all short labels
        first = raw.iloc[0].tolist()
        known = URL_NAMES | CONTENT_NAMES | CATEGORY_NAMES | EMBED_NAMES
        if any(_norm(v) in known for v in first) or (
                all(len(v) < 40 for v in first) and any(prof[c]['avg_len'] > 80 for c in range(ncols))):
            header_row = 0

    names = {c: _norm(raw.iat[header_row, c]) for c in range(ncols)} if header_row is not None else {}
    data = raw.iloc[(header_row + 1) if header_row is not None else 0:]
    prof = {c: _profile(data[c]) for c in range(ncols)}

    def by_name(options, exclude=()):
        for c in range(ncols):
            if c not in exclude and names.get(c) in options and prof[c]['fill'] > 0:
                return c
        return None

    embed_col = by_name(EMBED_NAMES)
    if embed_col is None:
        cands = [c for c in range(ncols) if prof[c]['embed'] > 0.5]
        embed_col = cands[0] if cands else None

    used = {embed_col}
    url_by_name = by_name(URL_NAMES, used)
    if url_by_name is not None and prof[url_by_name]['url'] >= 0.3:
        url_col = url_by_name
    if url_col is None:
        url_col = url_by_name
    used.add(url_col)

    content_col = by_name(CONTENT_NAMES, used)
    if content_col is None or prof[content_col]['avg_len'] < 20:
        cands = [c for c in range(ncols) if c not in used and prof[c]['url'] < 0.5 and prof[c]['embed'] < 0.5]
        content_col = max(cands, key=lambda c: prof[c]['avg_len'], default=None)
    used.add(content_col)

    category_col = by_name(CATEGORY_NAMES, used)
    if category_col is None and header_row is None:
        # Headerless file: a short, repeating text column is probably a category
        cands = [c for c in range(ncols) if c not in used and 0 < prof[c]['avg_len'] < 60
                 and prof[c]['url'] < 0.2 and prof[c]['numeric'] < 0.5 and prof[c]['unique'] < 0.6]
        category_col = cands[0] if cands else None

    return dict(header_row=header_row, url=url_col, content=content_col,
                category=category_col, embedding=embed_col)


def column_labels(raw, header_row):
    """Human labels like 'A · Address' (or 'A · https://example.com/...' when no header)."""
    labels = []
    for c in range(raw.shape[1]):
        if header_row is not None and raw.iat[header_row, c]:
            hint = raw.iat[header_row, c]
        else:
            start = header_row + 1 if header_row is not None else 0
            first = next((v for v in raw[c].tolist()[start:] if v), '')
            hint = first
        hint = (hint[:40] + '…') if len(hint) > 40 else hint
        labels.append(f"{col_letter(c)} · {hint}" if hint else col_letter(c))
    return labels


def build_dataframe(raw, mapping):
    """Standard DataFrame with URL, Content, optional Category / Embedding + skipped-row stats."""
    start = (mapping['header_row'] + 1) if mapping['header_row'] is not None else 0
    data = raw.iloc[start:].reset_index(drop=True)

    df = pd.DataFrame({'URL': data[mapping['url']], 'Content': data[mapping['content']]})
    if mapping.get('category') is not None:
        df['Category'] = data[mapping['category']].replace('', pd.NA)
    if mapping.get('embedding') is not None:
        df['Embedding'] = data[mapping['embedding']].replace('', pd.NA)

    total = len(df)
    no_url = df['URL'].str.strip() == ''
    no_content = (df['Content'].str.strip() == '') & ~no_url
    df = df[~no_url & ~no_content].reset_index(drop=True)
    stats = dict(total=total, no_url=int(no_url.sum()), no_content=int(no_content.sum()))
    return df, stats
