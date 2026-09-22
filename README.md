# Semantic Similarity Analyzer

A Streamlit app that measures semantic similarity between web pages using AI embeddings – find duplicate, near-duplicate and overlapping content.

## Features

- 🔍 **AI-Powered Analysis** - Uses OpenAI embeddings for semantic similarity
- 📊 **TF-IDF Support** - Optional keyword overlap analysis
- 🎯 **Similarity Levels** - Duplicate, Near-duplicate, Very similar, Similar, Related, Loosely related
- 📈 **Progress Tracking** - Real-time progress bars
- 💰 **Cost Estimation** - Know the cost before running
- 📥 **CSV Export** - Results, summary and embeddings as CSV files
- 🔄 **Reusable Embeddings** - Re-upload the embeddings CSV to skip API costs; embeddings are always downloadable, even if the analysis fails
- 🔒 **Privacy First** - No data stored on servers


## Usage

### Input File Format

Your CSV must have these columns:

**Required:**
- `URL` - The page URL
- `Content` - Full text content of the page

**Optional:**
- `Category` - Page category for grouping
- `Embedding` - Pre-computed embedding (JSON array). Rows without one are generated.

**Example:**
```csv
URL,Content,Category
https://example.com/page1,"Full page content here...",Blog
https://example.com/page2,"More content here...",Product
```

### Analysis Steps

1. **Upload CSV file**
2. **Enter OpenAI API key** (sidebar)
3. **Configure options:**
   - Enable/disable TF-IDF
   - Adjust similarity threshold
4. **Click "Start Analysis"**
5. **Download results:**
   - Data + Embeddings CSV
   - Similarity Results CSV
   - Summary CSV

### Output Files

- **Data + Embeddings CSV:** URL, Content, Category, Embedding – upload it again to reuse embeddings
- **Similarity Results CSV:** URL pairs, similarity level, semantic (and TF-IDF) similarity in %
- **Summary CSV:** totals per similarity level and key statistics

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
