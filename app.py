import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
from utils.embeddings import generate_embeddings, estimate_cost
from utils.similarity import calculate_similarities, categorize_priority
from utils.tfidf import calculate_tfidf_similarity

# Page config
st.set_page_config(
    page_title="Keyword Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Keyword Cannibalization Analyzer")
st.markdown("""
Analyze semantic similarity between your web pages to identify keyword cannibalization issues.
Upload your content, and we'll identify pages competing for the same search intent.
""")

# Sidebar - API Key and Options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your API key is only used during this session and is never stored."
    )
    
    if api_key:
        st.success("✅ API Key provided")
    else:
        st.warning("⚠️ Please enter your OpenAI API key")
    
    st.markdown("---")
    
    # Options
    st.header("📊 Analysis Options")
    
    include_tfidf = st.checkbox(
        "Include TF-IDF Analysis",
        value=False,
        help="Calculate keyword overlap in addition to semantic similarity. This is faster and free but less accurate."
    )
    
    similarity_threshold = st.slider(
        "AI Similarity Threshold (%)",
        min_value=75,
        max_value=95,
        value=85,
        step=5,
        help="Only show pairs with AI similarity above this threshold"
    ) / 100
    
    # Add separate TF-IDF threshold if TF-IDF is enabled
    if include_tfidf:
        tfidf_threshold = st.slider(
            "TF-IDF Similarity Threshold (%)",
            min_value=30,
            max_value=80,
            value=50,
            step=5,
            help="TF-IDF similarities are typically lower than AI similarities. 50% is a good starting point."
        ) / 100
    else:
        tfidf_threshold = 0.5  # Default value when not shown
    
    st.markdown("---")
    
    # Priority Legend
    st.header("🎯 Priority Levels")
    st.markdown("""
    - 🔴 **High Priority:** ≥ 95%
    - 🟡 **Medium Priority:** 90-95%
    - 🟢 **Low Priority:** 85-90%
    """)
    
    st.markdown("---")
    
    # Privacy Notice
    st.info("🔒 **Privacy:** Your data and API key are processed in-memory only and never stored on our servers.")

# Main content area
st.header("📤 Upload Your Data")

uploaded_file = st.file_uploader(
    "Upload CSV file with your content",
    type=['csv'],
    help="Required columns: URL, Content. Optional: Title, Category"
)

if uploaded_file:
    try:
        # Read the CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['URL', 'Content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Your CSV must have columns named: URL, Content (and optionally: Title, Category)")
            st.stop()
        
        # Check if optional columns exist
        has_categories = 'Category' in df.columns and not df['Category'].isna().all()
        has_titles = 'Title' in df.columns
        
        # Remove rows with empty content
        df = df[df['Content'].notna() & (df['Content'].str.strip() != '')]
        
        if len(df) == 0:
            st.error("❌ No valid content found in the file. Please check your Content column.")
            st.stop()
        
        if len(df) > 5000:
            st.error(f"❌ File too large: {len(df)} URLs found. Maximum supported: 5,000 URLs")
            st.info("Please split your data into smaller batches for analysis.")
            st.stop()
        
        # Display file info
        st.success(f"✅ File uploaded successfully: {len(df)} URLs found")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total URLs", len(df))
        with col2:
            metrics_text = []
            if has_titles:
                metrics_text.append("Title ✅")
            if has_categories:
                metrics_text.append("Category ✅")
            if not metrics_text:
                metrics_text.append("None")
            st.metric("Optional Fields", " | ".join(metrics_text))
        with col3:
            avg_content_length = df['Content'].str.len().mean()
            st.metric("Avg Content Length", f"{int(avg_content_length)} chars")
        
        # Show preview
        with st.expander("📋 Preview Data (first 5 rows)"):
            st.dataframe(df.head())
        
        # Cost estimation
        if api_key:
            st.header("💰 Cost Estimation")
            
            estimated_cost = estimate_cost(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated OpenAI Cost", f"${estimated_cost:.4f}")
            with col2:
                total_comparisons = (len(df) * (len(df) - 1)) // 2
                st.metric("Total Comparisons", f"{total_comparisons:,}")
            
            if include_tfidf:
                st.info("✅ TF-IDF analysis is free (no API costs)")
            
            st.markdown("---")
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                if not api_key:
                    st.error("❌ Please provide your OpenAI API key in the sidebar")
                    st.stop()
                
                # Start analysis
                st.header("⚙️ Processing...")
                
                # Step 1: Generate embeddings
                with st.spinner("Step 1/3: Generating AI embeddings..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    embeddings_result = generate_embeddings(
                        df, 
                        api_key,
                        progress_callback=lambda current, total: (
                            progress_bar.progress(current / total),
                            status_text.text(f"Processing {current}/{total} pages...")
                        )
                    )
                    
                    if embeddings_result['success']:
                        df['Embedding'] = embeddings_result['embeddings']
                        st.success(f"✅ Generated {len(embeddings_result['embeddings'])} embeddings")
                    else:
                        st.error(f"❌ Error: {embeddings_result['error']}")
                        st.stop()
                
                # Step 2: TF-IDF (optional)
                tfidf_results = None
                if include_tfidf:
                    with st.spinner("Step 2/3: Calculating TF-IDF similarity..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        tfidf_results = calculate_tfidf_similarity(
                            df,
                            threshold=tfidf_threshold,
                            progress_callback=lambda current, total: (
                                progress_bar.progress(current / total),
                                status_text.text(f"Comparing {current}/{total} pairs...")
                            )
                        )
                        st.success(f"✅ TF-IDF analysis complete: {len(tfidf_results)} pairs found above {tfidf_threshold*100}% threshold")
                else:
                    st.info("⏭️ Step 2/3: TF-IDF analysis skipped")
                
                # Step 3: Calculate similarities
                with st.spinner("Step 3/3: Calculating semantic similarities..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = calculate_similarities(
                        df,
                        threshold=similarity_threshold,
                        tfidf_results=tfidf_results,
                        include_tfidf=include_tfidf,
                        progress_callback=lambda current, total: (
                            progress_bar.progress(current / total),
                            status_text.text(f"Comparing {current}/{total} pairs...")
                        )
                    )
                    
                    st.success(f"✅ Found {len(results)} similar pairs above {similarity_threshold*100}% threshold")
                
                if len(results) == 0:
                    st.warning(f"⚠️ No similar pairs found above {similarity_threshold*100}% threshold. Try lowering the threshold.")
                    st.stop()
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Add priority categorization
                results_df['Priority'] = results_df['AI Similarity'].apply(categorize_priority)
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    high_priority = len(results_df[results_df['Priority'] == '🔴 High Priority'])
                    st.metric("High Priority", high_priority)
                with col2:
                    medium_priority = len(results_df[results_df['Priority'] == '🟡 Medium Priority'])
                    st.metric("Medium Priority", medium_priority)
                with col3:
                    low_priority = len(results_df[results_df['Priority'] == '🟢 Low Priority'])
                    st.metric("Low Priority", low_priority)
                with col4:
                    avg_similarity = results_df['AI Similarity'].mean()
                    st.metric("Avg Similarity", f"{avg_similarity*100:.1f}%")
                
                # Priority distribution chart
                st.subheader("📈 Priority Distribution")
                priority_counts = results_df['Priority'].value_counts()
                st.bar_chart(priority_counts)
                
                # Top 10 results preview
                st.subheader("🔝 Top 10 Similar Pairs")
                
                # Format for display
                display_df = results_df.head(10).copy()
                display_df['AI Similarity'] = display_df['AI Similarity'].apply(lambda x: f"{x*100:.2f}%")
                if include_tfidf:
                    display_df['TF-IDF Similarity'] = display_df['TF-IDF Similarity'].apply(
                        lambda x: f"{x*100:.2f}%" if x != "Not checked" else x
                    )
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download buttons
                st.header("⬇️ Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create Excel file
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Sheet 1: Original Data + Embeddings
                        export_df = df.copy()
                        export_df['AI Embedding'] = export_df['Embedding'].apply(
                            lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else str(x)
                        )
                        export_df = export_df.drop('Embedding', axis=1)
                        
                        # Reorder columns
                        cols = ['URL']
                        if has_titles:
                            cols.append('Title')
                        cols.extend(['Content'])
                        if has_categories:
                            cols.append('Category')
                        cols.append('AI Embedding')
                        
                        export_df = export_df[cols]
                        export_df.to_excel(writer, sheet_name='Original Data + Embeddings', index=False)
                        
                        # Sheet 2: Analysis Results
                        results_export = results_df.copy()
                        results_export.to_excel(writer, sheet_name='Similarity Analysis', index=False)
                        
                        # Sheet 3: Summary Statistics
                        summary_metrics = [
                            'Total URLs Analyzed',
                            'Total Similar Pairs Found',
                            'High Priority Pairs (≥95%)',
                            'Medium Priority Pairs (90-95%)',
                            'Low Priority Pairs (85-90%)',
                            '',
                            'Average AI Similarity',
                            'Maximum AI Similarity',
                            'Minimum AI Similarity',
                            '',
                            'AI Similarity Threshold Used',
                        ]
                        
                        summary_values = [
                            len(df),
                            len(results_df),
                            high_priority,
                            medium_priority,
                            low_priority,
                            '',
                            f"{results_df['AI Similarity'].mean()*100:.2f}%",
                            f"{results_df['AI Similarity'].max()*100:.2f}%",
                            f"{results_df['AI Similarity'].min()*100:.2f}%",
                            '',
                            f"{similarity_threshold*100}%",
                        ]
                        
                        if include_tfidf:
                            summary_metrics.append('TF-IDF Threshold Used')
                            summary_values.append(f"{tfidf_threshold*100}%")
                        
                        summary_metrics.extend([
                            'TF-IDF Analysis Included',
                            'Titles Provided',
                            'Categories Provided'
                        ])
                        
                        summary_values.extend([
                            'Yes' if include_tfidf else 'No',
                            'Yes' if has_titles else 'No',
                            'Yes' if has_categories else 'No'
                        ])
                        
                        summary_data = {
                            'Metric': summary_metrics,
                            'Value': summary_values
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    output.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="📥 Download Excel Results",
                        data=output,
                        file_name=f"cannibalization_analysis_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # Create JSON embeddings file
                    embeddings_dict = {}
                    for idx, row in df.iterrows():
                        url = row['URL']
                        embedding = row['Embedding']
                        if isinstance(embedding, np.ndarray):
                            embeddings_dict[url] = embedding.tolist()
                    
                    embeddings_json = json.dumps(embeddings_dict, indent=2)
                    
                    st.download_button(
                        label="📥 Download Embeddings JSON",
                        data=embeddings_json,
                        file_name=f"embeddings_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True,
                        help="Download embeddings for reuse in future analyses"
                    )
                
                st.success("✅ Analysis complete! Download your results above.")
        
        else:
            st.info("👆 Please enter your OpenAI API key in the sidebar to continue")
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.info("Please ensure your CSV has the correct format with 'URL' and 'Content' columns.")

else:
    # Instructions when no file uploaded
    st.info("👆 Upload your CSV file to begin analysis")
    
    st.markdown("### 📝 CSV Format Requirements")
    st.markdown("""
    Your CSV file must contain:
    - **URL** (required): The page URL
    - **Content** (required): The full text content of the page
    - **Title** (optional): The page title (can be empty)
    - **Category** (optional): Page category for grouping
    
    Example:
```
    URL,Title,Content,Category
    https://example.com/page1,"Page Title 1","Full page content here...",Blog
    https://example.com/page2,"","More content here...",Product
```
    """)
    
    st.markdown("### 🔑 API Key")
    st.markdown("""
    You'll need an OpenAI API key to use this tool. Get one at [platform.openai.com](https://platform.openai.com/api-keys).
    
    **Cost:** Approximately $0.02 per 1,000 pages analyzed.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by OpenAI • Your data is never stored</p>
</div>
""", unsafe_allow_html=True)
