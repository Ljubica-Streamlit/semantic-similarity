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
        help="Calculate keyword overlap for semantic similarity results."
    )
    
    # Show TF-IDF independently option only when TF-IDF is enabled
    if include_tfidf:
        show_tfidf_only_matches = st.checkbox(
            "Show Additional TF-IDF Matches",
            value=False,
            help="Also show pairs with high keyword overlap (TF-IDF) but low semantic similarity. Helps find pages using similar words but different meanings."
        )
    else:
        show_tfidf_only_matches = False
    
    similarity_threshold = st.slider(
        "Semantic Similarity Threshold (%)",
        min_value=75,
        max_value=95,
        value=85,
        step=5,
        help="Only show pairs with semantic similarity above this threshold"
    ) / 100
    
    # Add TF-IDF threshold slider if independent mode is enabled
    if include_tfidf and show_tfidf_only_matches:
        tfidf_threshold = st.slider(
            "TF-IDF Similarity Threshold (%)",
            min_value=30,
            max_value=95,
            value=85,
            step=5,
            help="Minimum TF-IDF similarity for additional matches"
        ) / 100
    else:
        tfidf_threshold = similarity_threshold  # Use same threshold as AI
    
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
    help="Required: URL, Content. Optional: Embedding (JSON array), Category"
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
            st.info("Your CSV must have columns named: URL, Content (and optionally: Embedding, Category)")
            st.stop()
        
        # Check if embeddings already exist in the file
        has_embeddings = 'Embedding' in df.columns or 'AI Embedding' in df.columns
        
        if has_embeddings:
            # Determine which column name is used
            embedding_col = 'Embedding' if 'Embedding' in df.columns else 'AI Embedding'
            
            # Parse embeddings from JSON strings to numpy arrays
            try:
                df['Embedding'] = df[embedding_col].apply(
                    lambda x: np.array(json.loads(x)) if pd.notna(x) and isinstance(x, str) else x
                )
                # Count valid embeddings
                valid_embeddings = df['Embedding'].apply(
                    lambda x: isinstance(x, np.ndarray) and len(x) > 0
                ).sum()
                
                if valid_embeddings > 0:
                    st.info(f"✅ Found {valid_embeddings} existing embeddings in file - will skip embedding generation!")
                else:
                    has_embeddings = False
                    st.warning("⚠️ Embedding column found but no valid embeddings detected")
            except Exception as e:
                has_embeddings = False
                st.warning(f"⚠️ Could not parse embeddings: {str(e)}")
        
        # Check if Category column exists
        has_categories = 'Category' in df.columns and not df['Category'].isna().all()
        
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
            st.metric("Has Embeddings", "Yes ✅" if has_embeddings else "No ❌")
        with col3:
            st.metric("Has Categories", "Yes ✅" if has_categories else "No ❌")
        
        # Show preview
        with st.expander("📋 Preview Data (first 5 rows)"):
            st.dataframe(df.head())
        
        # Cost estimation
        if api_key or has_embeddings:
            st.header("💰 Cost Estimation")
            
            if has_embeddings:
                st.success("✅ Using existing embeddings - No API cost for embedding generation!")
                estimated_cost = 0
            else:
                estimated_cost = estimate_cost(df)
            
            col1, col2 = st.columns(2)
            with col1:
                if has_embeddings:
                    st.metric("Estimated OpenAI Cost", "$0.00 (Using existing embeddings)")
                else:
                    st.metric("Estimated OpenAI Cost", f"${estimated_cost:.4f}")
            with col2:
                total_comparisons = (len(df) * (len(df) - 1)) // 2
                st.metric("Total Comparisons", f"{total_comparisons:,}")
            
            if include_tfidf:
                st.info("✅ TF-IDF analysis is free (no API costs)")
            
            st.markdown("---")
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                if not api_key and not has_embeddings:
                    st.error("❌ Please provide your OpenAI API key in the sidebar or upload a file with existing embeddings")
                    st.stop()
                
                # Start analysis
                st.header("⚙️ Processing...")
                
                # Step 1: Generate embeddings (only if needed)
                if not has_embeddings:
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
                else:
                    st.info("⏭️ Step 1/3: Using existing embeddings - skipped generation")
                
                # Step 2: Calculate semantic similarities
                with st.spinner("Step 2/3: Calculating semantic similarities..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = calculate_similarities(
                        df,
                        threshold=similarity_threshold,
                        progress_callback=lambda current, total: (
                            progress_bar.progress(current / total),
                            status_text.text(f"Comparing {current}/{total} pairs...")
                        )
                    )
                    
                    st.success(f"✅ Found {len(results)} pairs above {similarity_threshold*100}% semantic similarity threshold")
                
                # Step 3: TF-IDF analysis
                if include_tfidf:
                    with st.spinner("Step 3/3: Calculating TF-IDF for results..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Create list of URL pairs from semantic similarity results
                        semantic_pairs = [(r['URL 1'], r['URL 2']) for r in results]
                        
                        # Calculate TF-IDF only for semantic similarity pairs
                        from utils.tfidf import calculate_tfidf_for_pairs
                        tfidf_scores = calculate_tfidf_for_pairs(
                            df,
                            semantic_pairs,
                            progress_callback=lambda current, total: (
                                progress_bar.progress(current / total),
                                status_text.text(f"Calculating TF-IDF for {current}/{total} pairs...")
                            )
                        )
                        
                        # Add TF-IDF scores to results
                        for result in results:
                            key = tuple(sorted([result['URL 1'], result['URL 2']]))
                            result['TF-IDF Similarity'] = tfidf_scores.get(key, 0.0)
                        
                        # If show_tfidf_only_matches is enabled, find additional pairs
                        additional_pairs = []
                        if show_tfidf_only_matches:
                            status_text.text("Finding additional TF-IDF matches...")
                            
                            # Calculate TF-IDF for all pairs
                            all_tfidf_results = calculate_tfidf_similarity(
                                df,
                                threshold=tfidf_threshold,
                                progress_callback=None
                            )
                            
                            # Filter to only pairs NOT already in semantic results
                            existing_pairs = {tuple(sorted([r['URL 1'], r['URL 2']])) for r in results}
                            
                            for tfidf_result in all_tfidf_results:
                                pair_key = tuple(sorted([tfidf_result['URL 1'], tfidf_result['URL 2']]))
                                if pair_key not in existing_pairs:
                                    # This is a TF-IDF match that wasn't in semantic results
                                    # Get the semantic similarity for this pair
                                    url1, url2 = tfidf_result['URL 1'], tfidf_result['URL 2']
                                    
                                    # Find these URLs in the dataframe
                                    idx1 = df[df['URL'] == url1].index[0]
                                    idx2 = df[df['URL'] == url2].index[0]
                                    
                                    # Calculate semantic similarity for this pair
                                    from utils.similarity import calculate_cosine_similarity
                                    emb1 = df.loc[idx1, 'Embedding']
                                    emb2 = df.loc[idx2, 'Embedding']
                                    semantic_sim = calculate_cosine_similarity(emb1, emb2)
                                    
                                    additional_pairs.append({
                                        'URL 1': url1,
                                        'URL 2': url2,
                                        'Semantic Similarity': semantic_sim,
                                        'TF-IDF Similarity': tfidf_result['TF-IDF Similarity']
                                    })
                            
                            if additional_pairs:
                                st.info(f"✅ Found {len(additional_pairs)} additional pairs with high TF-IDF but low semantic similarity")
                        
                        st.success(f"✅ TF-IDF analysis complete")
                else:
                    st.info("⏭️ Step 3/3: TF-IDF analysis skipped")
                
                # Combine results
                if include_tfidf and show_tfidf_only_matches and additional_pairs:
                    # Add additional pairs to results
                    results.extend(additional_pairs)
                    st.info(f"📊 Total results: {len(results)} pairs ({len(results) - len(additional_pairs)} semantic + {len(additional_pairs)} TF-IDF only)")
                
                if len(results) == 0:
                    st.warning(f"⚠️ No similar pairs found above {similarity_threshold*100}% threshold. Try lowering the threshold.")
                    st.stop()
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Add priority categorization based on semantic similarity
                results_df['Priority'] = results_df['Semantic Similarity'].apply(categorize_priority)
                
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
                    avg_similarity = results_df['Semantic Similarity'].mean()
                    st.metric("Avg Semantic Similarity", f"{avg_similarity*100:.1f}%")
                
                # Priority distribution chart
                st.subheader("📈 Priority Distribution")
                priority_counts = results_df['Priority'].value_counts()
                st.bar_chart(priority_counts)
                
                # Top 10 results preview
                st.subheader("🔝 Top 10 Similar Pairs")
                
                # Show TF-IDF statistics if available
                if include_tfidf and 'TF-IDF Similarity' in results_df.columns:
                    tfidf_numeric = results_df['TF-IDF Similarity']
                    if len(tfidf_numeric) > 0:
                        st.info(f"📊 TF-IDF Stats: All {len(tfidf_numeric)} pairs analyzed (avg: {tfidf_numeric.mean()*100:.1f}%)")
                
                # Format for display
                display_df = results_df.head(10).copy()
                display_df['Semantic Similarity'] = display_df['Semantic Similarity'].apply(lambda x: f"{x*100:.2f}%")
                if include_tfidf and 'TF-IDF Similarity' in display_df.columns:
                    display_df['TF-IDF Similarity'] = display_df['TF-IDF Similarity'].apply(
                        lambda x: f"{x*100:.2f}%"
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
                        cols = ['URL', 'Content']
                        if has_categories:
                            cols.append('Category')
                        cols.append('AI Embedding')
                        
                        export_df = export_df[cols]
                        export_df.to_excel(writer, sheet_name='Original Data + Embeddings', index=False)
                        
                        # Sheet 2: Analysis Results
                        results_export = results_df.copy()
                        results_export.to_excel(writer, sheet_name='Similarity Analysis', index=False)
                        
                        # Sheet 3: Summary Statistics
                        summary_data = {
                            'Metric': [
                                'Total URLs Analyzed',
                                'Total Similar Pairs Found',
                                'High Priority Pairs (≥95%)',
                                'Medium Priority Pairs (90-95%)',
                                'Low Priority Pairs (85-90%)',
                                '',
                                'Average Semantic Similarity',
                                'Maximum Semantic Similarity',
                                'Minimum Semantic Similarity',
                                '',
                                'Semantic Similarity Threshold Used',
                                'TF-IDF Analysis Included',
                                'Embeddings Pre-existing',
                                'Categories Provided'
                            ],
                            'Value': [
                                len(df),
                                len(results_df),
                                high_priority,
                                medium_priority,
                                low_priority,
                                '',
                                f"{results_df['Semantic Similarity'].mean()*100:.2f}%",
                                f"{results_df['Semantic Similarity'].max()*100:.2f}%",
                                f"{results_df['Semantic Similarity'].min()*100:.2f}%",
                                '',
                                f"{similarity_threshold*100}%",
                                'Yes' if include_tfidf else 'No',
                                'Yes' if has_embeddings else 'No',
                                'Yes' if has_categories else 'No'
                            ]
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
    - **Embedding** (optional): Pre-computed AI embedding as JSON array - saves API costs!
    - **Category** (optional): Page category for grouping
    
    **Example - Basic:**
```csv
    URL,Content,Category
    https://example.com/page1,"Full page content here...",Blog
    https://example.com/page2,"More content here...",Product
```
    
    **Example - With Embeddings (no API key needed!):**
```csv
    URL,Content,Embedding,Category
    https://example.com/page1,"Content...","[0.123, -0.456, ...]",Blog
    https://example.com/page2,"Content...","[0.789, -0.012, ...]",Product
```
    """)
    
    st.markdown("### ✨ New Features")
    st.markdown("""
    - **Reuse Embeddings**: Upload files with existing embeddings to skip API costs
    - **Independent TF-IDF**: Find keyword cannibalization even when semantic similarity is low
    - **Flexible Analysis**: Mix and match AI and TF-IDF results for comprehensive insights
    """)
    
    st.markdown("### 🔑 API Key")
    st.markdown("""
    You'll need an OpenAI API key to use this tool. Get one at [platform.openai.com](https://platform.openai.com/api-keys).
    
    **Cost:** Approximately $0.02 per 1,000 pages analyzed.
    
    **Tip:** Download the embeddings from your first analysis to reuse them later without additional API costs!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Powered by OpenAI • Your data is never stored</p>
</div>
""", unsafe_allow_html=True)
