# Semantic Similarity Analyzer

A Streamlit app that measures semantic similarity between web pages using AI embeddings – find duplicate, near-duplicate and overlapping content, grouped into topics you can act on.

Best used together with your **keyword cannibalization audit**: this tool shows which pages overlap in content, and your keyword data shows which ones compete in search. Together they tell you whether to merge, redirect or keep both pages.

**[Open the app](https://semantic-similarity-ljubica.streamlit.app/)** · [About this tool](https://addonist.com/tools/semantic-similarity-analysis/)

## Features

- 🔍 **AI-Powered Analysis** - Uses OpenAI embeddings for semantic similarity
- 🧹 **Repeated blocks removed** - Nav, footer, cookie banners and CTAs are left out of the comparison
- 🗂️ **Topic groups** - Similar pages grouped by topic, likely duplicates first, with a column for your decision
- 🧱 **Template groups** - Thin pages that mainly share a template (e.g. location or event pages) grouped and named by the text they share
- 🧩 **Two scores per pair** - Unique-content similarity *and* full-page overlap
- 🎯 **Similarity Levels** - Duplicate, Near-duplicate, Very similar, Similar, Related, Loosely related
- 💰 **Cost Estimation** - See the cost before entering a key or running
- 📥 **One Excel download** - Every result in one file (CSV versions also available)
- 🔄 **Reusable Embeddings** - Re-upload your results to skip API costs; embeddings are always downloadable, even if the analysis stops early
- 🔒 **Privacy First** - No data or keys stored on our servers

## Usage

### Input File

Upload a **CSV or Excel (.xlsx)** file with page URLs and their content – **any column order, with or without headers**.

The app detects the columns automatically by what they contain:
- **URL** – the column that looks like web addresses (also `site.com/page` or `/blog/post`)
- **Content** – the column with the longest text
- **Category** (optional) – a column named e.g. Category, Type, Section, Cluster
- **Embedding** (optional) – from a previous run; only missing rows are generated

Comma, semicolon and tab separated files work, as do SEO-tool exports such as Screaming Frog
(`Address` / `Body Text`). If a guess is wrong, open **Change columns** and pick the right ones.
Empty rows and rows without a URL or content are skipped automatically.

### Analysis Steps

1. **Upload your CSV or Excel file**
2. **Check the cost estimate**, then **enter your OpenAI API key** in the sidebar (not needed if every page already has an embedding)
3. **Adjust options** in the sidebar if needed:
   - Similarity threshold (default 88%)
   - Repeated-block removal on/off, and on what share of pages a block must appear (default 80%)
4. **Click "Start Analysis"**
5. **Download** one Excel file with all results (CSV versions under *Prefer CSV files?*)

### Output

One Excel file (`semantic_similarity_<date>.xlsx`). Sheets appear when they have content:

| Sheet | What it holds |
|---|---|
| **Groups** | One row per similar page, grouped by topic. Start here. |
| **Template Groups** | One row per thin page that matches others mainly through a shared template |
| **Similarity Results** | Every similar pair at or above the threshold, with both scores |
| **Summary** | Totals, levels, pairs at each threshold, settings used |
| **Data + Embeddings** | Your pages and their embeddings – upload this file again to reuse them |

If the analysis stops early (e.g. the API key runs out of quota), the file still contains every embedding generated so far.

#### Groups

Every page in a group is at least as similar as the threshold to **every** other page in it, so unrelated pages can't chain together through a page in the middle. Groups are numbered by their closest pair, so likely duplicates come first.

| Column | Meaning |
|---|---|
| Group | Group number (empty: the page matches something but doesn't fit a whole group) |
| Group Cohesion | How alike the whole group is – the similarity level of its weakest link |
| Best Pair / Weakest Link | The highest and lowest similarity between two pages in the group |
| Closest Match | The page most similar to this one, and its score |
| Also Close To Group | The closest match sits in another group this page doesn't fully fit |
| Decision / Notes | For your verdict – Decision is a dropdown (Keep, Merge, Redirect, Noindex, Rewrite, Leave as is) |

The app doesn't suggest which page to keep – that depends on traffic and business data it doesn't have.

#### Template Groups

Thin pages (less than 40% unique content after repeated blocks are removed) that match another page mainly through shared text: at least 80% full-page overlap and 60% alike in meaning. They're grouped by the text left after site-wide blocks are removed – every page in a group shares at least 20% of it with every other – and each group is named by the block that sets it apart. Typical findings: location, event or contact pages built from one template with little of their own content.

### Overlap Types

Each similar pair also gets an overlap type, combining the two scores:

| Type | Meaning |
|---|---|
| Same content | The unique text matches too – a real duplicate |
| Same topic, different wording | Same subject, written differently |

### Similarity Levels

| Level | Semantic similarity |
|---|---|
| 🟥 Duplicate | 100% |
| 🔴 Near-duplicate | 98–99.9% |
| 🟠 Very similar | 95–98% |
| 🟡 Similar | 90–95% |
| 🟢 Related | 85–90% |
| ⚪ Loosely related | below 85% (only when you lower the threshold) |

The results also show how many pairs you'd get at 80, 85, 88, 90, 95 and 98%, so you can pick a threshold. Re-running with a different threshold reuses the embeddings and costs nothing.

## How the comparison works

1. **Repeated blocks are removed** (on by default). A block of text that appears on at least 80% of the
   uploaded pages is left out before embedding, so scores reflect what makes each page different.
   Needs at least 10 pages; the share is adjustable (50–100%); your file is never changed.
2. **Unique-content similarity** – OpenAI embeddings of the cleaned text, up to the model's input
   limit (~8,000 tokens per page). Longer pages are trimmed, and the app lists which ones.
3. **Full-page overlap** – word overlap on the original, untouched text. Free, no API calls.

Embeddings are tagged with the settings used (model, removal threshold, token cap). If you re-upload a file
whose embeddings were made with different settings, those embeddings are generated again, because scores
from different settings aren't comparable.

## Cost

- **OpenAI API:** ~$0.02 per 1,000 pages (`text-embedding-3-small`), shown before you run
- **Everything else:** Free (no API calls)

## Limits

- **Maximum:** 5,000 URLs per analysis
- **Threshold:** 70–99% similarity range
- **Excel:** cells hold up to 32,767 characters – longer pages are shortened in the Excel file only. To re-upload full text, use the *Data + Embeddings CSV*.

## Privacy & Security

- ✅ API keys used in-session only (never stored)
- ✅ All processing in-memory
- ✅ No data saved on our servers
- ✅ HTTPS encryption on Streamlit Cloud
- ℹ️ Page content is sent to OpenAI to create the embeddings, under your own API key and OpenAI's terms

## Support

Questions or feedback? [Message me on LinkedIn](https://www.linkedin.com/in/ljubica-popovic-lp/), or create an issue on GitHub.

## Author

Built by [Ljubica Popovic](https://www.linkedin.com/in/ljubica-popovic-lp/).

## License

MIT License – see [LICENSE](LICENSE). Copyright (c) 2026 Ljubica Popovic.
