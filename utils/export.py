import json
import re
import numpy as np
import pandas as pd

# Control characters that break spreadsheet apps (tab, newline, CR are kept)
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def clean_text(value):
    if isinstance(value, str):
        return _CONTROL_CHARS.sub(' ', value)
    return value


def embedding_to_str(x, decimals=8):
    if isinstance(x, np.ndarray) and x.size > 0:
        return json.dumps([round(float(v), decimals) for v in x])
    return ''


def to_csv_bytes(df):
    """CSV with UTF-8 BOM so Excel opens accents/emoji correctly."""
    clean = df.copy()
    for col in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = clean[col].map(clean_text)
    return clean.to_csv(index=False).encode('utf-8-sig')


def embeddings_csv(df, has_categories):
    """URL, Content, (Category), Embedding – can be re-uploaded to skip API costs."""
    out = pd.DataFrame({'URL': df['URL'], 'Content': df['Content']})
    if has_categories:
        out['Category'] = df['Category']
    out['Embedding'] = df['Embedding'].apply(embedding_to_str)
    return to_csv_bytes(out)


def results_csv(results_df):
    out = results_df.copy()
    for col in ('Semantic Similarity', 'TF-IDF Similarity'):
        if col in out.columns:
            out[col] = (out[col].astype(float) * 100).round(2)
    out = out.rename(columns={'Semantic Similarity': 'Semantic Similarity (%)',
                              'TF-IDF Similarity': 'TF-IDF Similarity (%)'})
    return to_csv_bytes(out)


def summary_csv(rows):
    return to_csv_bytes(pd.DataFrame(rows, columns=['Metric', 'Value']))
