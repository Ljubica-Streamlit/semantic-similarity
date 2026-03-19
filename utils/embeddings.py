import numpy as np
from openai import OpenAI
import time

def estimate_cost(df):
    """Estimate OpenAI API cost for embeddings"""
    # Average tokens per page (rough estimate)
    avg_tokens = 1000
    total_tokens = len(df) * avg_tokens
    
    # text-embedding-3-small pricing: $0.02 per 1M tokens
    cost = (total_tokens / 1_000_000) * 0.02
    
    return cost

def generate_embeddings(df, api_key, progress_callback=None):
    """Generate embeddings for all content with retry logic and partial results support"""
    try:
        client = OpenAI(api_key=api_key)
        embeddings = []
        skipped_rows = []
        
        total = len(df)
        max_retries = 3
        base_delay = 1  # seconds
        
        for count, (idx, row) in enumerate(df.iterrows()):
            # Combine title and content
            content = str(row['Content'])[:8000]  # Limit to 8k chars
            
            # Create text for embedding
            text_for_embedding = f"page content: {content}"
            
            success = False
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text_for_embedding
                    )
                    
                    embedding = response.data[0].embedding
                    embeddings.append(np.array(embedding))
                    success = True
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if this is an insufficient quota error (won't resolve with retries)
                    if 'insufficient_quota' in error_str:
                        return {
                            'success': False,
                            'error': f"OpenAI quota exceeded at row {count + 1}/{total}. Please check your billing at platform.openai.com.",
                            'partial_embeddings': embeddings,
                            'skipped_rows': skipped_rows,
                            'processed_count': count
                        }
                    
                    # For other errors (rate limit, transient), retry with exponential backoff
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                        time.sleep(delay)
            
            if not success:
                # All retries failed for this row — skip it
                embeddings.append(None)
                skipped_rows.append({
                    'row': count + 1,
                    'url': row.get('URL', f'Row {count + 1}'),
                    'error': str(last_error)
                })
            
            # Update progress — use count (sequential) and clamp to 1.0
            if progress_callback:
                progress_value = min((count + 1) / total, 1.0)
                progress_callback(count + 1, total, progress_value)
            
            # Small delay between successful requests to avoid rate limits
            if success:
                time.sleep(0.15)
        
        return {
            'success': True,
            'embeddings': embeddings,
            'skipped_rows': skipped_rows,
            'processed_count': total
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"API Error: {str(e)}",
            'partial_embeddings': embeddings if 'embeddings' in dir() else [],
            'skipped_rows': skipped_rows if 'skipped_rows' in dir() else [],
            'processed_count': 0
        }
