import numpy as np
import pandas as pd
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
    """Generate embeddings for all content"""
    try:
        client = OpenAI(api_key=api_key)
        embeddings = []
        
        total = len(df)
        
        for idx, row in df.iterrows():
            # Get content
            content = str(row['Content'])[:8000]  # Limit to 8k chars
            
            # Create text for embedding
            # Include title if available
            if 'Title' in df.columns and pd.notna(row['Title']) and str(row['Title']).strip():
                title = str(row['Title']).strip()
                text_for_embedding = f"page title: {title}\n\npage content: {content}"
            else:
                text_for_embedding = f"page content: {content}"
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text_for_embedding
                )
                
                embedding = response.data[0].embedding
                embeddings.append(np.array(embedding))
                
                # Update progress
                if progress_callback:
                    progress_callback(idx + 1, total)
                
                # Small delay to avoid rate limits
                time.sleep(0.05)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Error generating embedding for row {idx}: {str(e)}"
                }
        
        return {
            'success': True,
            'embeddings': embeddings
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"API Error: {str(e)}"
        }
