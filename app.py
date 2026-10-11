import hashlib
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils.embeddings import generate_embeddings, estimate_cost, MODEL, MAX_TOKENS
from utils.similarity import (
    calculate_similarities, count_pairs_at, calculate_cosine_similarity, categorize_similarity, classify_overlap,
    SIMILARITY_LEVELS, LEVEL_ORDER, TEMPLATE_TWIN_MIN, TEMPLATE_TWIN_MIN_SEMANTIC, MAX_TEMPLATE_TWIN_PAIRS,
)
from utils import boilerplate
from utils.tfidf import build_matrix, calculate_tfidf_for_pairs, calculate_tfidf_similarity
from utils.groups import group_pages, groups_frame, pair_groups, DECISIONS
from utils.templates import group_twins, page_blocks, shared_text, templates_frame, TEMPLATE_SHARED_MIN
from utils.export import (
    embeddings_csv, results_csv, groups_csv, templates_csv, summary_csv, excel_workbook, embeddings_frame, results_frame,
    count_over_cell_limit, looks_cut_by_excel, EXCEL_CELL_LIMIT,
)
from utils.columns import load_table, detect_columns, column_labels, build_dataframe

st.set_page_config(page_title="Semantic Similarity Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Similarity Analyzer")
st.markdown("""
Measure how semantically similar your pages are using AI embeddings.
Find duplicate, near-duplicate and overlapping content across your site.
""")


# ---------------------------------------------------------------- helpers
THRESHOLD_STEPS = (0.80, 0.85, 0.88, 0.90, 0.95, 0.98)  # rows of the 'pairs at each threshold' table


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

    similarity_threshold = st.slider(
        "Semantic Similarity Threshold (%)", min_value=70, max_value=99, value=88, step=1,
        help="Only show pairs whose unique content is at least this similar") / 100

    strip_boilerplate = st.checkbox(
        "Remove repeated blocks before comparing", value=True,
        help="Nav, footer, cookie banners and CTAs that appear on most pages are left out of the "
             "comparison, so scores reflect the unique content. Your file is never changed.")
    boilerplate_threshold = boilerplate.DEFAULT_THRESHOLD
    if strip_boilerplate:
        boilerplate_threshold = st.slider(
            "Remove blocks found on at least (% of pages)", min_value=50, max_value=100, value=80, step=5,
            help="Lower this to also remove blocks shared by one section only (e.g. all case studies).") / 100

    st.markdown("---")
    st.header("🎯 Similarity Levels")
    st.markdown("\n".join(f"- {label}: {desc}" for _, label, desc in SIMILARITY_LEVELS))

    st.markdown("---")
    st.info("🔒 **Privacy:** Your data and API key are processed in-memory only and never stored on our servers.")


# ---------------------------------------------------------------- upload
st.header("📤 Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file with your page URLs and their content",
    type=['csv', 'xlsx', 'tsv', 'txt'],
    help="Any column order, with or without headers. Columns are detected automatically.")

if not uploaded_file:
    st.session_state.pop('run', None)
    st.info("👆 Upload a CSV or Excel file with page URLs and their content – any column order, "
            "with or without headers.")
    st.markdown("""
**That's it.** The app finds the URL and content columns by itself. Optional extras are picked up
automatically too: a **Category** column, and an **Embedding** column from a previous run
(re-upload your *Data + Embeddings CSV* to skip API costs – only missing rows are generated).

You'll need an OpenAI API key ([get one here](https://platform.openai.com/api-keys)) –
about **$0.02 per 1,000 pages**.
""")
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666'>Built with Streamlit • Powered by OpenAI • "
                "Your data is never stored</div>", unsafe_allow_html=True)
    st.stop()

# Reset stored results when a different file is uploaded
raw_bytes = uploaded_file.getvalue()
file_id = hashlib.md5(raw_bytes).hexdigest()
if st.session_state.get('file_id') != file_id:
    st.session_state['file_id'] = file_id
    st.session_state.pop('run', None)

try:
    raw = load_table(uploaded_file.name, raw_bytes)
except Exception as e:
    st.error(f"❌ Could not open this file: {e}")
    st.info("Please upload a CSV or Excel (.xlsx) file.")
    st.stop()

if raw.empty or raw.shape[1] < 2:
    st.error("❌ The file needs at least two columns: page URLs and their content.")
    st.stop()

detected = detect_columns(raw)
NONE = "— none —"

# Column confirmation (auto-detected, user can override)
with st.expander("🔧 Change columns", expanded=detected['url'] is None or detected['content'] is None):
    has_header = st.checkbox("First row contains column names", value=detected['header_row'] is not None,
                             key=f"hdr_{file_id}")
    header_row = (detected['header_row'] if detected['header_row'] is not None else 0) if has_header else None
    labels = column_labels(raw, header_row)
    idx = list(range(raw.shape[1]))

    def pick(label, key, default, optional=False):
        options = ([None] if optional else []) + idx
        fmt = lambda c: NONE if c is None else labels[c]
        start = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=start, format_func=fmt, key=f"{key}_{file_id}")

    c1, c2 = st.columns(2)
    with c1:
        url_col = pick("URL column", "url", detected['url'])
        cat_col = pick("Category column (optional)", "cat", detected['category'], optional=True)
    with c2:
        content_col = pick("Content column", "content",
                           detected['content'] if detected['content'] is not None else (1 if raw.shape[1] > 1 else 0))
        emb_col = pick("Embedding column (optional)", "emb", detected['embedding'], optional=True)

if url_col == content_col:
    st.error("❌ URL and Content can't be the same column – open **Change columns** above to fix it.")
    st.stop()

mapping = dict(header_row=header_row, url=url_col, content=content_col, category=cat_col,
               embedding=emb_col, settings=detected.get('settings'))

# Results from a run with different columns no longer describe this data
mapping_key = (header_row, url_col, content_col, cat_col, emb_col)
if st.session_state.get('mapping_key') != mapping_key:
    st.session_state['mapping_key'] = mapping_key
    st.session_state.pop('run', None)

df, row_stats = build_dataframe(raw, mapping)
labels = column_labels(raw, header_row)

detected_text = f"URL = **{labels[url_col]}**, Content = **{labels[content_col]}**"
if cat_col is not None:
    detected_text += f", Category = **{labels[cat_col]}**"
if emb_col is not None:
    detected_text += f", Embeddings = **{labels[emb_col]}**"
st.success(f"✅ Detected columns: {detected_text}")

skipped = row_stats['no_url'] + row_stats['no_content']
if skipped:
    parts = []
    if row_stats['no_url']:
        parts.append(f"{row_stats['no_url']} without a URL")
    if row_stats['no_content']:
        parts.append(f"{row_stats['no_content']} without content")
    st.caption(f"ℹ️ {skipped} rows skipped ({', '.join(parts)}).")

if len(df) == 0:
    st.error("❌ No rows with both a URL and content were found. Open **Change columns** above and check the selection.")
    st.stop()
if len(df) < 2:
    st.error("❌ At least 2 pages are needed to compare.")
    st.stop()
if len(df) > 5000:
    st.error(f"❌ File too large: {len(df)} URLs found. Maximum supported: 5,000 URLs")
    st.stop()

# ---------------------------------------------------------------- repeated blocks
raw_texts = df['Content'].tolist()
if strip_boilerplate:
    clean_texts, bp_stats = boilerplate.analyse(raw_texts, threshold=boilerplate_threshold)
else:
    clean_texts, bp_stats = list(raw_texts), {'applied': False, 'reason': 'Switched off.', 'heavy': False,
                                              'unique_pct': [100.0] * len(df)}
df['Clean Content'] = clean_texts
df['Unique Content %'] = bp_stats.get('unique_pct', [100.0] * len(df))
if bp_stats['applied']:
    df['Page Note'] = [boilerplate.thin_page_note(p, t) for p, t in zip(df['Unique Content %'], clean_texts)]

settings_label = (f"model={MODEL}; boilerplate="
                  f"{('on@' + str(int(boilerplate_threshold * 100)) + '%') if bp_stats['applied'] else 'off'}; "
                  f"max_tokens={MAX_TOKENS}")

# Existing embeddings (tolerant: bad rows just become "missing")
embedding_col = 'Embedding' if 'Embedding' in df.columns else None
df['Embedding'] = df['Embedding'].apply(parse_embedding) if embedding_col else [None] * len(df)

# Embeddings made with other settings would give quietly wrong scores – regenerate them
mismatched_settings, mismatched_count = set(), 0
if 'Embedding Settings' in df.columns:
    previous = df['Embedding Settings'].fillna('').astype(str).str.strip()
    stale = df['Embedding'].apply(lambda e: isinstance(e, np.ndarray)) & (previous != '') & (previous != settings_label)
    mismatched_settings = set(previous[stale])
    mismatched_count = int(stale.sum())
    df['Embedding'] = [None if s else e for e, s in zip(df['Embedding'], stale)]

existing_count = count_valid(df['Embedding'])
missing_count = len(df) - existing_count
has_categories = 'Category' in df.columns and not df['Category'].isna().all()

duplicate_urls = int(df['URL'].duplicated().sum())
cut_by_excel = int(df['Content'].map(looks_cut_by_excel).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Pages", len(df))
c2.metric("Existing Embeddings", f"{existing_count}/{len(df)}")
c3.metric("Has Categories", "Yes ✅" if has_categories else "No")
if duplicate_urls:
    st.warning(f"⚠️ {duplicate_urls} duplicate URLs found in the file – they will be compared like separate pages.")
if mismatched_count:
    st.warning(f"⚠️ {mismatched_count} embeddings in this file were made with different settings "
               f"({'; '.join(sorted(mismatched_settings))}) and can't be compared with the current ones "
               f"({settings_label}). They will be generated again. To reuse them instead, match the settings "
               f"in the sidebar.")
elif embedding_col and 0 < existing_count < len(df):
    st.info(f"ℹ️ {missing_count} rows have no embedding – only those will be generated.")
if cut_by_excel:
    st.warning(f"⚠️ {cut_by_excel} pages were cut to Excel's {EXCEL_CELL_LIMIT:,}-character cell limit in an "
               f"earlier export. Their embeddings are fine, but Full-page Overlap is measured on the shortened "
               f"text. Upload the *Data + Embeddings CSV* instead to keep the full content.")

with st.expander("📋 Preview Data (first 5 rows)"):
    st.dataframe(df.drop(columns=['Embedding', 'Clean Content']).head())

if bp_stats['applied']:
    st.subheader("🧹 Repeated blocks")
    c1, c2, c3 = st.columns(3)
    c1.metric("Blocks removed", bp_stats['removed_block_count'])
    c2.metric("Characters removed per page", f"{bp_stats['removed_chars_avg']:,}")
    c3.metric("Unique content left", f"{bp_stats['avg_unique_pct']}%")
    with st.expander("See what is being removed"):
        st.dataframe(pd.DataFrame([{'On % of pages': b['share'], 'Text': b['text']}
                                   for b in bp_stats['removed_blocks']]),
                     hide_index=True, width="stretch")
        st.caption("Only the text sent for comparison changes – your file and the exported content stay complete.")
    if bp_stats['heavy']:
        st.warning(f"⚠️ The repeated blocks make up {100 - bp_stats['avg_unique_pct']:.0f}% of the average page. "
                   f"These pages may be template variants (e.g. location or product pages) whose shared text is "
                   f"the finding itself. You can continue – pairs that match mainly through shared blocks are "
                   f"listed in the **Template Groups** sheet – or raise the threshold / switch removal off in the sidebar.")
    thin_count = int(df['Page Note'].astype(bool).sum())
    if thin_count:
        st.caption(f"ℹ️ {thin_count} pages have less than {boilerplate.THIN_PAGE_MAX_UNIQUE_PCT}% unique content "
                   f"left after removal. They are marked as mostly shared template in the **Page Note** column.")
elif strip_boilerplate:
    st.caption(f"🧹 Repeated-block removal not applied: {bp_stats['reason']}")

# ---------------------------------------------------------------- cost
st.header("💰 Cost Estimation")
c1, c2 = st.columns(2)
missing_texts = [t for t, e in zip(clean_texts, df['Embedding']) if not isinstance(e, np.ndarray)]
cost = estimate_cost(missing_texts) if missing_count else 0.0
c1.metric("Estimated OpenAI Cost",
          "$0.00 (Using existing embeddings)" if missing_count == 0
          else "< $0.0001" if cost < 0.0001 else f"${cost:.4f}")
c2.metric("Total Comparisons", f"{(len(df) * (len(df) - 1)) // 2:,}")
st.markdown("---")

# Shown after the cost, so people see the price before deciding to enter a key
if missing_count and not api_key:
    st.info("👆 Please enter your OpenAI API key in the sidebar to continue")
    st.stop()


# ---------------------------------------------------------------- analysis
def run_analysis(df):
    """Runs every step. Stores progress in session_state so nothing paid for is lost."""
    run = {'df': df, 'results_df': None, 'errors': [], 'warnings': [], 'new_embeddings': 0,
           'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
           'settings': dict(threshold=similarity_threshold, boilerplate=bp_stats['applied'],
                            boilerplate_threshold=boilerplate_threshold, label=settings_label),
           'boilerplate': bp_stats}
    st.session_state['run'] = run

    # Step 1 – embeddings (built from the text with repeated blocks removed)
    if missing_count:
        progress = st.progress(0.0, text="Step 1/3: Generating AI embeddings...")
        was_missing = [not isinstance(e, np.ndarray) for e in df['Embedding']]
        emb = generate_embeddings(
            df['Clean Content'].tolist(), df['URL'].astype(str).tolist(), api_key,
            existing=df['Embedding'].tolist(),
            progress_callback=lambda done, total, p: progress.progress(p, text=f"Step 1/3: Embedding {done}/{total} pages..."))
        df['Embedding'] = emb['embeddings']
        # Label only the rows made now – uploaded rows keep their own settings record
        settings = df['Embedding Settings'].fillna('').astype(str).tolist() \
            if 'Embedding Settings' in df.columns else [''] * len(df)
        df['Embedding Settings'] = [settings_label if m and isinstance(e, np.ndarray) else s
                                    for m, e, s in zip(was_missing, df['Embedding'], settings)]
        run['new_embeddings'] = emb['new_count']
        if emb.get('trimmed'):
            run['warnings'].append(f"{emb['trimmed']} pages were longer than the model's limit and were trimmed "
                                   f"to their first ~{MAX_TOKENS:,} tokens.")
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

    # Step 2 – unique-content similarity
    try:
        progress = st.progress(0.0, text="Step 2/3: Comparing unique content...")
        results = calculate_similarities(
            df, threshold=similarity_threshold,
            progress_callback=lambda cur, tot: progress.progress(min(cur / tot, 1.0), text=f"Step 2/3: Compared {cur}/{tot} pages..."))
    except Exception as e:
        run['errors'].append(f"Similarity calculation failed: {e}")
        run['trace'] = traceback.format_exc()
        return
    try:
        steps = sorted(set(THRESHOLD_STEPS) | {similarity_threshold})
        run['pair_counts'] = count_pairs_at(df, steps)
    except Exception as e:
        run['warnings'].append(f"Pairs per threshold could not be counted: {e}")

    # Step 3 – full-page overlap on the raw text (free, no API)
    twins = []
    try:
        progress = st.progress(0.0, text="Step 3/3: Measuring full-page overlap...")
        tfidf_matrix = build_matrix(df['Content'].fillna('').astype(str).tolist(), filter_common=False)
        pairs = [(r['URL 1'], r['URL 2']) for r in results]
        overlap = calculate_tfidf_for_pairs(df, pairs, tfidf_matrix=tfidf_matrix) if pairs else {}
        for r in results:
            r['Full-page Overlap'] = overlap.get(tuple(sorted([r['URL 1'], r['URL 2']])), 0.0)

        # Pages that match mainly through shared blocks would otherwise disappear – they get their own sheet.
        if bp_stats['applied']:
            known = set(overlap.keys())
            emb_by_url, thin = {}, {}
            for u, e, pct in zip(df['URL'], df['Embedding'], df['Unique Content %']):
                emb_by_url.setdefault(u, e)
                thin.setdefault(u, boilerplate.is_thin(pct))
            extra = 0
            for t in calculate_tfidf_similarity(df, threshold=TEMPLATE_TWIN_MIN, tfidf_matrix=tfidf_matrix):
                key = tuple(sorted([t['URL 1'], t['URL 2']]))
                if key in known or not (thin.get(t['URL 1']) and thin.get(t['URL 2'])):
                    continue
                e1, e2 = emb_by_url.get(t['URL 1']), emb_by_url.get(t['URL 2'])
                sem = (calculate_cosine_similarity(e1, e2)
                       if isinstance(e1, np.ndarray) and isinstance(e2, np.ndarray) else np.nan)
                # Shared blocks alone (e.g. the same form on a webinar and a contact page) don't make twins.
                # No URL or folder rule here: real twins can live in different directories.
                if pd.notna(sem) and sem < TEMPLATE_TWIN_MIN_SEMANTIC:
                    continue
                extra += 1
                if extra > MAX_TEMPLATE_TWIN_PAIRS:
                    run['warnings'].append(
                        f"More than {MAX_TEMPLATE_TWIN_PAIRS:,} template-twin pairs were found – only the "
                        f"first {MAX_TEMPLATE_TWIN_PAIRS:,} are listed.")
                    break
                twins.append({'URL 1': t['URL 1'], 'URL 2': t['URL 2'], 'Semantic Similarity': sem,
                              'Full-page Overlap': t['TF-IDF Similarity']})
        progress.progress(1.0, text="Step 3/3: Full-page overlap done")
    except Exception as e:
        run['warnings'].append(f"Full-page overlap could not be calculated: {e}")

    unique_pct = dict(zip(df['URL'].astype(str), df['Unique Content %']))
    notes = dict(zip(df['URL'].astype(str), df['Page Note'])) if 'Page Note' in df.columns else None
    cats = dict(zip(df['URL'].astype(str), df['Category'])) if has_categories else None
    if twins:
        try:
            run['templates_df'] = build_templates(df, twins, unique_pct, notes, cats)
        except Exception as e:
            run['warnings'].append(f"Template groups could not be built: {e}")
            run['trace'] = traceback.format_exc()

    if results:
        results_df = pd.DataFrame(results)
        results_df.insert(2, 'Similarity Level', results_df['Semantic Similarity'].apply(categorize_similarity))
        results_df.insert(3, 'Overlap Type', [
            classify_overlap(r['Semantic Similarity'], r.get('Full-page Overlap'), similarity_threshold)
            for _, r in results_df.iterrows()])
        results_df['Unique Content % 1'] = results_df['URL 1'].map(unique_pct)
        results_df['Unique Content % 2'] = results_df['URL 2'].map(unique_pct)
        if notes is not None:
            results_df['Page Note 1'] = results_df['URL 1'].map(notes)
            results_df['Page Note 2'] = results_df['URL 2'].map(notes)
        results_df = results_df.sort_values('Semantic Similarity', ascending=False, na_position='last')

        # Topic groups from the pairs (template twins are a different finding, in their own sheet)
        main = results_df
        if len(main):
            try:
                emb_by_url = {}
                for u, e in zip(df['URL'].astype(str), df['Embedding']):
                    emb_by_url.setdefault(u, e)
                pages = sorted(set(main['URL 1']) | set(main['URL 2']))
                membership, group_stats = group_pages(pages, [emb_by_url[u] for u in pages], similarity_threshold)
                results_df.insert(4, 'Group', pair_groups(results_df, membership))
                run['groups_df'] = groups_frame(main, membership, group_stats, unique_pct, notes, cats)
            except Exception as e:
                run['warnings'].append(f"Topic groups could not be built: {e}")
        run['results_df'] = results_df.reset_index(drop=True)


def build_templates(df, twins, unique_pct, notes, cats):
    """Template Groups sheet: grouped by the text left after site-wide blocks are removed."""
    first = {}
    for i, u in enumerate(df['URL'].astype(str)):
        first.setdefault(u, i)
    pages = sorted({t['URL 1'] for t in twins} | {t['URL 2'] for t in twins})
    rows = [first[u] for u in pages]
    clean = df['Clean Content'].fillna('').astype(str).tolist()
    remaining = build_matrix(clean, filter_common=False)  # IDF from the whole site
    shared = (remaining[rows] @ remaining[rows].T).toarray()  # rows are L2-normalised
    membership = group_twins(pages, shared)

    blocks = [page_blocks(t) for t in clean]
    site_counts = {}
    for page in blocks:
        for key in page:
            site_counts[key] = site_counts.get(key, 0) + 1
    members = {}
    for url, group in membership.items():
        members.setdefault(group, []).append(first[url])
    names = {g: shared_text([blocks[i] for i in idx], site_counts, len(clean)) for g, idx in members.items()}
    return templates_frame(twins, membership, names, unique_pct, notes, cats)


if st.button("🚀 Start Analysis", type="primary", width="stretch"):
    try:
        run_analysis(df.copy())
    except Exception as e:
        run = st.session_state.setdefault('run', {'df': df, 'results_df': None, 'errors': [], 'warnings': [],
                                                   'new_embeddings': 0, 'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                                                   'settings': dict(threshold=similarity_threshold, boilerplate=bp_stats['applied'],
                                                                    boilerplate_threshold=boilerplate_threshold,
                                                                    label=settings_label)})
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

    results_df = run['results_df']
    groups_df = run.get('groups_df')
    templates_df = run.get('templates_df')
    has_groups = groups_df is not None and len(groups_df) > 0
    has_templates = templates_df is not None and len(templates_df) > 0
    data_sheet = ('Data + Embeddings', embeddings_frame(rdf, has_categories))

    summary_rows = [
        ('Total URLs Analyzed', len(rdf)),
        ('URLs With Embeddings', n_emb),
        ('New Embeddings Generated', run['new_embeddings']),
    ]
    if results_df is not None:
        level_counts = results_df['Similarity Level'].value_counts()
        summary_rows += [('Similar Pairs Found', len(results_df))] + \
            [(label, int(level_counts.get(label, 0))) for label in LEVEL_ORDER]
    if has_groups:
        grouped = groups_df['Group'].notna()
        summary_rows += [
            ('Topic Groups', int(groups_df.loc[grouped, 'Group'].nunique())),
            ('Pages in Topic Groups', int(grouped.sum())),
            ('Similar Pages Not in a Group', int((~grouped).sum())),
        ]
    if has_templates:
        in_template = templates_df['Template Group'].notna()
        summary_rows += [
            ('Template Groups', int(templates_df.loc[in_template, 'Template Group'].nunique())),
            ('Pages in Template Groups', int(in_template.sum())),
            ('Template Pages Not in a Group', int((~in_template).sum())),
        ]
    if results_df is not None:
        scores = results_df['Semantic Similarity']
        summary_rows += [
            ('Average Semantic Similarity', fmt_pct(scores.mean())),
            ('Maximum Semantic Similarity', fmt_pct(scores.max())),
            ('Minimum Semantic Similarity', fmt_pct(scores.min())),
        ]
    summary_rows += [
        ('Similarity Threshold Used', f"{run['settings']['threshold'] * 100:.0f}%"),
    ] + [(f"Pairs at {c * 100:.0f}% or above", n) for c, n in run.get('pair_counts', {}).items()] + [
        ('Repeated Blocks Removed', 'Yes' if run['settings']['boilerplate'] else 'No'),
        ('Removal Threshold', f"{run['settings']['boilerplate_threshold'] * 100:.0f}% of pages"
         if run['settings']['boilerplate'] else 'n/a'),
        ('Embedding Settings', run['settings']['label']),
    ] + ([(f"Overlap type: {k}", int(v)) for k, v in results_df['Overlap Type'].value_counts().items()]
         if results_df is not None and 'Overlap Type' in results_df.columns else [])
    summary_sheet = ('Summary', pd.DataFrame(summary_rows, columns=['Metric', 'Value']).astype(str))
    has_findings = results_df is not None or has_templates

    # One Excel file with every sheet. Embeddings are always included – the user paid for them.
    st.header("⬇️ Download")
    if run['errors'] and n_emb:
        st.info(f"💾 Your {n_emb} embeddings are saved in the file below. Upload it again to continue "
                f"without paying for them twice.")

    sheets = ([('Groups', results_frame(groups_df))] if has_groups else []) + \
        ([('Template Groups', results_frame(templates_df))] if has_templates else []) + \
        ([('Similarity Results', results_frame(results_df))] if results_df is not None else []) + \
        [summary_sheet, data_sheet]
    excel_ok = False
    if n_emb or has_findings:
        try:
            label = "📥 Download Results (Excel)" if has_findings else \
                f"📥 Download Embeddings (Excel, {n_emb}/{len(rdf)} pages)"
            dropdowns = {'Groups': {'Decision': DECISIONS}, 'Template Groups': {'Decision': DECISIONS}}
            st.download_button(
                label=label, data=excel_workbook(sheets, dropdowns=dropdowns),
                file_name=f"semantic_similarity_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", width="stretch", key="dl_excel", on_click="ignore",
                help="One file: " + ", ".join(name for name, _ in sheets) + ". "
                     "Re-upload it later to reuse the embeddings.")
            st.caption("Sheets: " + " · ".join(name for name, _ in sheets))
            too_long = count_over_cell_limit(rdf['Content'])
            if too_long:
                st.caption(f"ℹ️ {too_long} pages are longer than Excel's {EXCEL_CELL_LIMIT:,}-character cell limit "
                           f"and are shortened in this file. To re-upload with the full text, use the "
                           f"*Data + Embeddings CSV* below.")
            excel_ok = True
        except Exception as e:
            st.warning(f"⚠️ Could not build the Excel file ({e}) – download the CSV files below instead.")

    # CSV versions: tucked away normally, shown openly if Excel failed
    with st.expander("Prefer CSV files?", expanded=not excel_ok):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if n_emb:
                st.download_button("Data + Embeddings CSV", data=embeddings_csv(rdf, has_categories),
                                   file_name=f"embeddings_{ts}.csv", mime="text/csv",
                                   width="stretch", key="dl_embeddings", on_click="ignore")
        if has_groups:
            with c2:
                st.download_button("Groups CSV", data=groups_csv(groups_df),
                                   file_name=f"similarity_groups_{ts}.csv", mime="text/csv",
                                   width="stretch", key="dl_groups", on_click="ignore")
        if has_templates:
            with c3:
                st.download_button("Template Groups CSV", data=templates_csv(templates_df),
                                   file_name=f"similarity_templates_{ts}.csv", mime="text/csv",
                                   width="stretch", key="dl_templates", on_click="ignore")
        if results_df is not None:
            with c4:
                st.download_button("Similarity Results CSV", data=results_csv(results_df),
                                   file_name=f"similarity_results_{ts}.csv", mime="text/csv",
                                   width="stretch", key="dl_results", on_click="ignore")
        if has_findings:
            with c5:
                st.download_button("Summary CSV", data=summary_csv(summary_rows),
                                   file_name=f"similarity_summary_{ts}.csv", mime="text/csv",
                                   width="stretch", key="dl_summary", on_click="ignore")

    def show_templates():
        st.subheader("🧱 Template Groups")
        in_group = templates_df['Template Group'].notna()
        t1, t2, t3 = st.columns(3)
        t1.metric("Template groups", int(templates_df.loc[in_group, 'Template Group'].nunique()))
        t2.metric("Pages in template groups", int(in_group.sum()))
        t3.metric("Template pages not in a group", int((~in_group).sum()))
        st.caption(f"Thin pages (less than {boilerplate.THIN_PAGE_MAX_UNIQUE_PCT}% unique content) that match mainly "
                   f"through text they share. Pages are grouped by the text left after site-wide blocks are "
                   f"removed: every page in a group shares at least {TEMPLATE_SHARED_MIN * 100:.0f}% of it with "
                   f"every other. *Shared Text* is the block that sets the group apart – common in the group, "
                   f"rare elsewhere on the site. Record verdicts in the Template Groups sheet.")
        shown = templates_df.drop(columns=['Decision', 'Notes']).copy()
        for col in ('Full-page Overlap', 'Semantic Similarity'):
            shown[col] = shown[col].apply(fmt_pct)
        st.dataframe(shown, width="stretch", hide_index=True, height=400)

    if results_df is None:
        if has_templates:
            st.info(f"ℹ️ No page pairs at or above {run['settings']['threshold'] * 100:.0f}% – but some pages share "
                    f"a template. See **Template Groups** below, or lower the threshold to look for similar pairs.")
            show_templates()
        elif not run['errors']:
            st.warning(f"⚠️ No similar pairs found at or above {run['settings']['threshold'] * 100:.0f}%. "
                       f"Try lowering the threshold.")
        st.stop()

    st.header("📊 Analysis Results")
    shown_levels = [l for l in LEVEL_ORDER if level_counts.get(l, 0) > 0]
    cols = st.columns(max(len(shown_levels), 1) + 1)
    for col, label in zip(cols, shown_levels):
        col.metric(label, int(level_counts[label]))
    cols[-1].metric("Avg Similarity", fmt_pct(results_df['Semantic Similarity'].mean()))

    if 'Overlap Type' in results_df.columns:
        st.subheader("🧩 What kind of overlap")
        type_counts = results_df['Overlap Type'].value_counts()
        tcols = st.columns(max(len(type_counts), 1))
        for col, (label, count) in zip(tcols, type_counts.items()):
            col.metric(label, int(count))
        st.caption("**Same content** – the unique text matches too. "
                   "**Same topic, different wording** – same subject, written differently. "
                   "Pages that match mainly through shared template text are in **Template Groups** below.")

    st.subheader("📈 Similarity Level Distribution")
    dist = pd.DataFrame({'Pairs': [int(level_counts.get(l, 0)) for l in LEVEL_ORDER]}, index=LEVEL_ORDER)
    st.bar_chart(dist[dist['Pairs'] > 0])

    if run.get('pair_counts'):
        st.subheader("🎚️ Pairs at Each Threshold")
        current = run['settings']['threshold']
        st.dataframe(pd.DataFrame([{'Threshold': f"{c * 100:.0f}%" + ("  ← used" if abs(c - current) < 1e-9 else ""),
                                    'Similar pairs': n} for c, n in run['pair_counts'].items()]),
                     hide_index=True)
        st.caption("Change the threshold in the sidebar and run again to see a different cut – existing "
                   "embeddings are reused, so it costs nothing.")

    if has_groups:
        st.subheader("🗂️ Topic Groups")
        grouped = groups_df['Group'].notna()
        g1, g2, g3 = st.columns(3)
        g1.metric("Groups", int(groups_df.loc[grouped, 'Group'].nunique()))
        g2.metric("Pages in groups", int(grouped.sum()))
        g3.metric("Similar pages not in a group", int((~grouped).sum()))
        st.caption(f"Every page in a group is at least {run['settings']['threshold'] * 100:.0f}% similar to every "
                   f"other page in it. Groups are sorted by *Best Pair* (their closest two pages), so likely "
                   f"duplicates come first; *Group Cohesion* rates the *Weakest Link* – how alike the whole group "
                   f"is. *Also Close To Group* flags a page whose closest match sits in another group it doesn't "
                   f"fully fit. Pages with no Group match something but no whole group. Record verdicts in "
                   f"**Decision** and **Notes** in the Groups sheet.")
        shown = groups_df.drop(columns=['Decision', 'Notes']).copy()
        for col in ('Best Pair', 'Weakest Link', 'Closest Match Score'):
            shown[col] = shown[col].apply(fmt_pct)
        st.dataframe(shown, width="stretch", hide_index=True, height=400)

    if has_templates:
        show_templates()

    st.subheader("🔝 Top 10 Most Similar Pairs")
    st.caption("\"Semantic Similarity\" compares the unique content; \"Full-page Overlap\" compares the raw text.")
    display_df = results_df.head(10).copy()
    display_df['Semantic Similarity'] = display_df['Semantic Similarity'].apply(fmt_pct)
    if 'Full-page Overlap' in display_df.columns:
        display_df['Full-page Overlap'] = display_df['Full-page Overlap'].apply(fmt_pct)
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.success("✅ Analysis complete! Download your results above.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666'>Built with Streamlit • Powered by OpenAI • "
            "Your data is never stored</div>", unsafe_allow_html=True)
