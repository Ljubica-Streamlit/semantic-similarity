# Keyword Cannibalization Analyzer

A Streamlit app that analyzes semantic similarity between web pages to identify keyword cannibalization issues using AI embeddings.

## Features

- 🔍 **AI-Powered Analysis** - Uses OpenAI embeddings for semantic similarity
- 📊 **TF-IDF Support** - Optional keyword overlap analysis
- 🎯 **Priority Levels** - Automatic categorization (High/Medium/Low)
- 📈 **Progress Tracking** - Real-time progress bars
- 💰 **Cost Estimation** - Know the cost before running
- 📥 **Excel Export** - Download results with multiple sheets
- 🔄 **Reusable Embeddings** - Save embeddings for future use
- 🔒 **Privacy First** - No data stored on servers

## Installation

### Local Development

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd keyword-cannibalization-tool
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the app:**
```bash
streamlit run app.py
```

5. **Open browser:**
Navigate to `http://localhost:8501`

## Deployment to Streamlit Cloud

### 1. **Prepare GitHub Repository**
```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Create private repository on GitHub
# Then push:
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. **Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

### 3. **Keep Code Private, App Public**

- ✅ Repository: Set to **Private** on GitHub
- ✅ Streamlit App: Will be **Public** (anyone can access)
- ✅ Code: Remains **Private** (only you can see)

## Usage

### Input File Format

Your CSV must have these columns:

**Required:**
- `URL` - The page URL
- `Content` - Full text content of the page

**Optional:**
- `Category` - Page category for grouping

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
   - Excel file with 3 sheets
   - JSON file with embeddings

### Output Files

**Excel file contains:**
- **Sheet 1:** Original data + AI embeddings
- **Sheet 2:** Similarity analysis results
- **Sheet 3:** Summary statistics

**JSON file contains:**
- Embeddings for each URL (reusable)

## Cost

- **OpenAI API:** ~$0.02 per 1,000 pages
- **TF-IDF:** Free (no API calls)

## Limits

- **Maximum:** 5,000 URLs per analysis
- **Threshold:** 75-95% similarity range

## Privacy & Security

- ✅ API keys used in-session only (never stored)
- ✅ All processing in-memory
- ✅ No data saved on servers
- ✅ HTTPS encryption on Streamlit Cloud

## Support

For issues or questions, create an issue on GitHub.

## License

MIT License
