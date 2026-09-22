import hashlib
import io
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils.embeddings import generate_embeddings, estimate_cost
from utils.similarity import (
    calculate_similarities, calculate_cosine_similarity, categorize_similarity,
    SIMILARITY_LEVELS, LEVEL_ORDER,
)
from utils.tfidf import calculate_tfidf_for_pairs, calculate_tfidf_similarity
from utils.export import embeddings_csv, results_csv, summary_csv

st.set_page_config(page_title="Semantic Similarity Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Similarity Analyzer")
st.markdown("""
Measure how semantically similar your pages are using AI embeddings.
Find duplicate, near-duplicate and overlapping content across your site.
""")


# ---------------------------------------------------------------- helpers
def read_csv_any_encoding(uploaded):
    raw = uploaded.getvalue()
    last_err = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def parse_embedding(x):
    """Return np.ndarray or None. Never raises."""
    if isinstance(x, np.ndarray):
        return x if x.size else None
    if isinstance(x, str) and x.strip().startswith('['):
        try:
            arr = np.array(json.loads(x), dtype=np.float32)
            return arr if arr.ndim == 1 and arr.size else None
        except Exception:
            return None
    return None


def count_valid(series):
    return int(series.apply(lambda e: isinstance(e, np.ndarray) and e.size > 0).sum())


def fmt_pct(x):
    return f"{x * 100:.2f}%" if pd.notna(x) else "–"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                            help="Your API key is only used during this session and is never stored.")
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.info("Enter your OpenAI API key (not needed if your file already has embeddings)")

    st.markdown("---")
    st.header("📊 Analysis Options")

    include_tfidf = st.checkbox("Include TF-IDF Analysis", value=False,
                                help="Calculate keyword overlap for semantic similarity results.")
    show_tfidf_only_matches = False
    if include_tfidf:
        show_tfidf_only_matches = st.checkbox(
            "Show Additional TF-IDF Matches", value=False,
            help="Also show pairs with high keyword overlap (TF-IDF) but lower semantic similarity.")

    similarity_threshold = st.slider(
        "Semantic Similarity Threshold (%)", min_value=70, max_value=99, value=85, step=1,
        help="Only show pairs with semantic similarity at or above this value") / 100

    tfidf_threshold = similarity_threshold
    if include_tfidf and show_tfidf_only_matches:
        tfidf_threshold = st.slider("TF-IDF Similarity Threshold (%)", min_value=30, max_value=95,
                                    value=85, step=5) / 100

    st.markdown("---")
    st.header("🎯 Similarity Levels")
    st.markdown("\n".join(f"- {label}: {desc}" for _, label, desc in SIMILARITY_LEVELS))

    st.markdown("---")
    st.info("🔒 **Privacy:** Your data and API key are processed in-memory only and never stored on our servers.")


# ---------------------------------------------------------------- upload
st.header("📤 Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload CSV file with your content", type=['csv'],
    help="Required: URL, Content. Optional: Embedding (JSON array), Category")

if not uploaded_file:
    st.session_state.pop('run', None)
    st.info("👆 Upload your CSV file to begin analysis")
    st.markdown("### 📝 CSV Format Requirements")
    st.markdown("""
Your CSV file must contain:
- **URL** (required): The page URL
- **Content** (required): The full text content of the page
- **Embedding** (optional): Pre-computed embedding as JSON array – saves API costs!
- **Category** (optional): Page category for grouping

**Tip:** The *Data + Embeddings CSV* you download after an analysis can be uploaded again as-is –
no new API cost. If some rows are missing embeddings, only those rows are generated.
""")
    st.markdown("### 🔑 API Key")
    st.markdown("""
You'll need an OpenAI API key. Get one at [platform.openai.com](https://platform.openai.com/api-keys).

**Cost:** approximately $0.02 per 1,000 pages.
""")
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666'>Built with Streamlit • Powered by OpenAI • "
                "Your data is never stored</div>", unsafe_allow_html=True)
    st.stop()

# Reset stored results when a different file is uploaded
file_id = hashlib.md5(uploaded_file.getvalue()).hexdigest()
if st.session_state.get('file_id') != file_id:
    st.session_state['file_id'] = file_id
    st.session_state.pop('run', None)

try:
    df = read_csv_any_encoding(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read the CSV file: {e}")
    st.info("Please make sure the file is a valid CSV with 'URL' and 'Content' columns.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
missing_columns = [c for c in ('URL', 'Content') if c not in df.columns]
if missing_columns:
    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
    st.info("Your CSV must have columns named: URL, Content (and optionally: Embedding, Category)")
    st.stop()

df['URL'] = df['URL'].astype(str).str.strip()
df['Content'] = df['Content'].where(df['Content'].notna(), '').astype(str)
df = df[df['Content'].str.strip() != ''].reset_index(drop=True)

if len(df) == 0:
    st.error("❌ No valid content found in the file. Please check your Content column.")
    st.stop()
if len(df) > 5000:
    st.error(f"❌ File too large: {len(df)} URLs found. Maximum supported: 5,000 URLs")
    st.stop()

# Existing embeddings (tolerant: bad rows just become "missing")
embedding_col = 'Embedding' if 'Embedding' in df.columns else ('AI Embedding' if 'AI Embedding' in df.columns else None)
df['Embedding'] = df[embedding_col].apply(parse_embedding) if embedding_col else [None] * len(df)
existing_count = count_valid(df['Embedding'])
missing_count = len(df) - existing_count
has_categories = 'Category' in df.columns and not df['Category'].isna().all()

duplicate_urls = int(df['URL'].duplicated().sum())

st.success(f"✅ File uploaded successfully: {len(df)} URLs found")
c1, c2, c3 = st.columns(3)
c1.metric("Total URLs", len(df))
c2.metric("Existing Embeddings", f"{existing_count}/{len(df)}")
c3.metric("Has Categories", "Yes ✅" if has_categories else "No ❌")
if duplicate_urls:
    st.warning(f"⚠️ {duplicate_urls} duplicate URLs found in the file – they will be compared like separate pages.")
if embedding_col and 0 < existing_count < len(df):
    st.info(f"ℹ️ {missing_count} rows have no embedding – only those will be generated.")

with st.expander("📋 Preview Data (first 5 rows)"):
    st.dataframe(df.drop(columns=['Embedding']).head())

if missing_count and not api_key:
    st.info("👆 Please enter your OpenAI API key in the sidebar to continue")
    st.stop()

# ---------------------------------------------------------------- cost
st.header("💰 Cost Estimation")
c1, c2 = st.columns(2)
c1.metric("Estimated OpenAI Cost",
          "$0.00 (Using existing embeddings)" if missing_count == 0 else f"${estimate_cost(df, missing_count):.4f}")
c2.metric("Total Comparisons", f"{(len(df) * (len(df) - 1)) // 2:,}")
if include_tfidf:
    st.info("✅ TF-IDF analysis is free (no API costs)")
st.markdown("---")


# ---------------------------------------------------------------- analysis
def run_analysis(df):
    """Runs every step. Stores progress in session_state so nothing paid for is lost."""
    run = {'df': df, 'results_df': None, 'errors': [], 'warnings': [], 'new_embeddings': 0,
           'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
           'settings': dict(threshold=similarity_threshold, tfidf=include_tfidf)}
    st.session_state['run'] = run

    # Step 1 – embeddings
    if missing_count:
        progress = st.progress(0.0, text="Step 1/3: Generating AI embeddings...")
        emb = generate_embeddings(
            df, api_key, existing=df['Embedding'].tolist(),
            progress_callback=lambda done, total, p: progress.progress(p, text=f"Step 1/3: Embedding {done}/{total} pages..."))
        df['Embedding'] = emb['embeddings']
        run['new_embeddings'] = emb['new_count']
        if emb['skipped_rows']:
            run['warnings'].append(
                f"{len(emb['skipped_rows'])} pages could not be embedded and were left out: " +
                ", ".join(s['url'] for s in emb['skipped_rows'][:20]) +
                (" ..." if len(emb['skipped_rows']) > 20 else ""))
        if emb['error']:
            run['errors'].append(f"Embedding stopped: {emb['error']}")
            return
    else:
        st.info("⏭️ Step 1/3: Using existing embeddings – skipped generation")

    # Step 2 – semantic similarity
    try:
        progress = st.progress(0.0, text="Step 2/3: Calculating semantic similarities...")
        results = calculate_similarities(
            df, threshold=similarity_threshold,
            progress_callback=lambda cur, tot: progress.progress(min(cur / tot, 1.0), text=f"Step 2/3: Compared {cur}/{tot} pages..."))
    except Exception as e:
        run['errors'].append(f"Similarity calculation failed: {e}")
        run['trace'] = traceback.format_exc()
        return

    # Step 3 – TF-IDF (optional; failure here never discards semantic results)
    if include_tfidf and results:
        try:
            progress = st.progress(0.0, text="Step 3/3: Calculating TF-IDF...")
            pairs = [(r['URL 1'], r['URL 2']) for r in results]
            scores = calculate_tfidf_for_pairs(
                df, pairs,
                progress_callback=lambda cur, tot: progress.progress(min(cur / tot, 1.0), text=f"Step 3/3: TF-IDF {cur}/{tot} pairs..."))
            for r in results:
                r['TF-IDF Similarity'] = scores.get(tuple(sorted([r['URL 1'], r['URL 2']])), 0.0)

            if show_tfidf_only_matches:
                existing_pairs = set(scores.keys())
                emb_by_url = {}
                for u, e in zip(df['URL'], df['Embedding']):
                    emb_by_url.setdefault(u, e)
                for t in calculate_tfidf_similarity(df, threshold=tfidf_threshold):
                    key = tuple(sorted([t['URL 1'], t['URL 2']]))
                    if key in existing_pairs:
                        continue
                    e1, e2 = emb_by_url.get(t['URL 1']), emb_by_url.get(t['URL 2'])
                    sem = calculate_cosine_similarity(e1, e2) if (isinstance(e1, np.ndarray) and isinstance(e2, np.ndarray)) else np.nan
                    results.append({'URL 1': t['URL 1'], 'URL 2': t['URL 2'],
                                    'Semantic Similarity': sem, 'TF-IDF Similarity': t['TF-IDF Similarity'],
                                    'Match Type': 'TF-IDF only'})
        except Exception as e:
            run['warnings'].append(f"TF-IDF analysis failed and was skipped: {e}")

    if results:
        results_df = pd.DataFrame(results)
        results_df.insert(2, 'Similarity Level',
                          results_df['Semantic Similarity'].apply(lambda s: categorize_similarity(s) if pd.notna(s) else '–'))
        run['results_df'] = results_df


if st.button("🚀 Start Analysis", type="primary", width="stretch"):
    try:
        run_analysis(df.copy())
    except Exception as e:
        run = st.session_state.setdefault('run', {'df': df, 'results_df': None, 'errors': [], 'warnings': [],
                                                   'new_embeddings': 0, 'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")})
        run['errors'].append(f"Unexpected error: {e}")
        run['trace'] = traceback.format_exc()


# ---------------------------------------------------------------- results (persist across reruns)
run = st.session_state.get('run')
if run:
    rdf = run['df']
    ts = run['timestamp']
    n_emb = count_valid(rdf['Embedding'])

    for w in run['warnings']:
        st.warning(f"⚠️ {w}")
    for e in run['errors']:
        st.error(f"❌ {e}")
    if run.get('trace'):
        with st.expander("Technical details"):
            st.code(run['trace'])

    # Embeddings are always offered first – the user paid for them
    st.header("⬇️ Download")
    if n_emb:
        if run['errors']:
            st.info(f"💾 Your {n_emb} embeddings are saved below. Upload this CSV again to continue "
                    f"without paying for them twice.")
        try:
            st.download_button(
                label=f"📥 Data + Embeddings CSV ({n_emb}/{len(rdf)} pages)",
                data=embeddings_csv(rdf, has_categories),
                file_name=f"embeddings_{ts}.csv", mime="text/csv",
                width="stretch", key="dl_embeddings",
                help="Re-upload this file later to skip embedding costs")
        except Exception as e:
            st.error(f"❌ Could not build the embeddings file: {e}")

    results_df = run['results_df']
    if results_df is None:
        if not run['errors']:
            st.warning(f"⚠️ No similar pairs found at or above {similarity_threshold * 100:.0f}%. "
                       f"Try lowering the threshold.")
        st.stop()

    level_counts = results_df['Similarity Level'].value_counts()
    summary_rows = [
        ('Total URLs Analyzed', len(rdf)),
        ('URLs With Embeddings', n_emb),
        ('New Embeddings Generated', run['new_embeddings']),
        ('Similar Pairs Found', len(results_df)),
    ] + [(label, int(level_counts.get(label, 0))) for label in LEVEL_ORDER] + [
        ('Average Semantic Similarity', fmt_pct(results_df['Semantic Similarity'].mean())),
        ('Maximum Semantic Similarity', fmt_pct(results_df['Semantic Similarity'].max())),
        ('Minimum Semantic Similarity', fmt_pct(results_df['Semantic Similarity'].min())),
        ('Similarity Threshold Used', f"{run['settings']['threshold'] * 100:.0f}%"),
        ('TF-IDF Analysis Included', 'Yes' if run['settings']['tfidf'] else 'No'),
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Similarity Results CSV", data=results_csv(results_df),
                           file_name=f"similarity_results_{ts}.csv", mime="text/csv",
                           width="stretch", key="dl_results")
    with c2:
        st.download_button("📥 Summary CSV", data=summary_csv(summary_rows),
                           file_name=f"similarity_summary_{ts}.csv", mime="text/csv",
                           width="stretch", key="dl_summary")

    st.header("📊 Analysis Results")
    shown_levels = [l for l in LEVEL_ORDER if level_counts.get(l, 0) > 0]
    cols = st.columns(max(len(shown_levels), 1) + 1)
    for col, label in zip(cols, shown_levels):
        col.metric(label, int(level_counts[label]))
    cols[-1].metric("Avg Similarity", fmt_pct(results_df['Semantic Similarity'].mean()))

    st.subheader("📈 Similarity Level Distribution")
    dist = pd.DataFrame({'Pairs': [int(level_counts.get(l, 0)) for l in LEVEL_ORDER]}, index=LEVEL_ORDER)
    st.bar_chart(dist[dist['Pairs'] > 0])

    st.subheader("🔝 Top 10 Most Similar Pairs")
    display_df = results_df.head(10).copy()
    display_df['Semantic Similarity'] = display_df['Semantic Similarity'].apply(fmt_pct)
    if 'TF-IDF Similarity' in display_df.columns:
        display_df['TF-IDF Similarity'] = display_df['TF-IDF Similarity'].apply(fmt_pct)
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.success("✅ Analysis complete! Download your results above.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>Built with Streamlit • Powered by OpenAI • "
            "Your data is never stored</div>", unsafe_allow_html=True)
