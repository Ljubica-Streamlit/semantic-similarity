# Semantic Similarity Analyzer

A Streamlit app that measures semantic similarity between web pages using AI embeddings – find duplicate, near-duplicate and overlapping content.

## Features

- 🔍 **AI-Powered Analysis** - Uses OpenAI embeddings for semantic similarity
- 📊 **TF-IDF Support** - Optional keyword overlap analysis
- 🎯 **Similarity Levels** - Duplicate, Near-duplicate, Very similar, Similar, Related, Loosely related
- 📈 **Progress Tracking** - Real-time progress bars
- 💰 **Cost Estimation** - Know the cost before running
- 📥 **One Excel download** - Results, summary and embeddings in one file (CSV versions also available)
- 🔄 **Reusable Embeddings** - Re-upload the embeddings CSV to skip API costs; embeddings are always downloadable, even if the analysis fails
- 🔒 **Privacy First** - No data stored on servers


## Usage

### Input File

Upload a **CSV or Excel (.xlsx)** file with page URLs and their content – **any column order, with or without headers**.

The app detects the columns automatically by what they contain:
- **URL** – the column that looks like web addresses (also `site.com/page` or `/blog/post`)
- **Content** – the column with the longest text
- **Category** (optional) – a column named e.g. Category, Type, Section, Cluster
- **Embedding** (optional) – JSON arrays from a previous run; only missing rows are generated

Comma, semicolon and tab separated files work, as do SEO-tool exports such as Screaming Frog
(`Address` / `Body Text`). If a guess is wrong, open **Change columns** and pick the right ones.
Empty rows and rows without a URL or content are skipped automatically.

### Analysis Steps

1. **Upload your CSV or Excel file**
2. **Enter OpenAI API key** (sidebar)
3. **Configure options:**
   - Enable/disable TF-IDF
   - Adjust similarity threshold
4. **Click "Start Analysis"**
5. **Download results:**
   - One Excel file with 3 sheets (CSV versions available too)

### Output Files

One Excel file (`semantic_similarity_<date>.xlsx`) with three sheets:
- **Similarity Results:** URL pairs, similarity level, semantic (and TF-IDF) similarity in %
- **Summary:** totals per similarity level and key statistics
- **Data + Embeddings:** URL, Content, Category, Embedding – upload the file again to reuse embeddings

If the analysis stops early (e.g. OpenAI quota), the file still contains every embedding generated so far.
The same data is available as separate CSV files under *Prefer CSV files?*.

### Similarity Levels

| Level | Semantic similarity |
|---|---|
| 🟥 Duplicate | 100% |
| 🔴 Near-duplicate | 98–99.9% |
| 🟠 Very similar | 95–98% |
| 🟡 Similar | 90–95% |
| 🟢 Related | 85–90% |
| ⚪ Loosely related | below 85% |

## Cost

- **OpenAI API:** ~$0.02 per 1,000 pages
- **TF-IDF:** Free (no API calls)

## Limits

- **Maximum:** 5,000 URLs per analysis
- **Threshold:** 70-99% similarity range

## Privacy & Security

- ✅ API keys used in-session only (never stored)
- ✅ All processing in-memory
- ✅ No data saved on servers
- ✅ HTTPS encryption on Streamlit Cloud

## Support

For issues or questions, create an issue on GitHub.

## License

MIT License
