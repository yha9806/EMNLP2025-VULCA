import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import torch


def load_text_from_file_id(file_id_str: str, base_data_path: Path, append_txt_if_missing: bool = False):
    """Loads text content from a file path constructed from file_id_str and base_data_path."""
    if not file_id_str or pd.isna(file_id_str):
        # print(f"Warning: Received empty or NaN file_id. Skipping text loading.")
        return None
    
    processed_file_id_str = str(file_id_str) # Ensure it's a string for path operations

    if append_txt_if_missing:
        if not processed_file_id_str.lower().endswith('.txt'):
            processed_file_id_str = f"{processed_file_id_str}.txt"
            # print(f"[DEBUG] Appended .txt, new file_id_str for loading: {processed_file_id_str}")

    try:
        # Assuming file_id_str is like "FolderName/actual_file_name.txt"
        # or just "actual_file_name.txt" if it's directly in a known subfolder not captured by file_id_str structure.
        # For this project, file_id structure is typically FolderName/filename.txt
        # e.g., 乔迅（Jonathan Hay）_清初中国的绘画与现代性/1.1_石涛画像.txt
        full_text_path = base_data_path / processed_file_id_str
        # print(f"[DEBUG] Attempting to load file_id: '{processed_file_id_str}' from path: {full_text_path}")
        
        if not full_text_path.exists():
            print(f"[DEBUG] File NOT FOUND for file_id: '{processed_file_id_str}' at resolved path: {full_text_path.resolve()}")
            return None
        
        with open(full_text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading text for file_id '{processed_file_id_str}': {e} at path {full_text_path}")
        return None


def extract_author_name(file_id_part):
    """Extracts author name from a file_id part (assumed to be folder name)."""
    if isinstance(file_id_part, str):
        # 假设作者名是下划线之前的部分
        return file_id_part.split('_', 1)[0]
    return "Unknown"


def process_input_source(df_input: pd.DataFrame, base_text_path: Path, source_name: str, id_col_for_text_loading: str,
                          author_col_name_if_direct=None, id_col_for_author_extraction=None, model=None):
    """
    Process a single input source DataFrame, extracting text, author labels, and generating embeddings.
    
    Parameters:
    - df_input: Input DataFrame
    - base_text_path: Base directory for loading text files
    - source_name: Name of the source ('human_expert' or 'mllm')
    - id_col_for_text_loading: Column name containing the identifier for loading text files
    - author_col_name_if_direct: Column containing direct author information (for MLLMs)
    - id_col_for_author_extraction: Column for extracting author name (for human expert)
    - model: SentenceTransformer model for generating embeddings
    
    Returns:
    - DataFrame with processed data including embeddings
    """
    print(f"\nProcessing {source_name} data from DataFrame (shape: {df_input.shape})...")
    
    df = df_input.copy() # Work on a copy
    
    if df.empty:
        print(f"Input DataFrame for {source_name} is empty. Skipping.")
        return pd.DataFrame()
    
    # --- MODIFICATION: Determine actual ID column for text loading and if suffix is needed for MLLM ---
    actual_id_col_for_text_loading = id_col_for_text_loading
    mllm_append_txt_suffix = False

    if source_name == 'mllm':
        # Check if the primary id_col_for_text_loading (e.g., 'original_relative_file_path') is usable
        if id_col_for_text_loading not in df.columns or df[id_col_for_text_loading].isnull().all():
            if 'file_id' in df.columns and not df['file_id'].isnull().all():
                print(f"INFO: For MLLM source '{source_name}', primary ID column '{id_col_for_text_loading}' is missing or empty. "
                      f"Falling back to 'file_id' column and will append '.txt' suffix if needed.")
                actual_id_col_for_text_loading = 'file_id'
                mllm_append_txt_suffix = True
            else:
                print(f"ERROR: For MLLM source '{source_name}', neither '{id_col_for_text_loading}' nor 'file_id' are usable for text loading. Skipping.")
                return pd.DataFrame()
        elif actual_id_col_for_text_loading not in df.columns: # Should be caught above, but defensive check
            print(f"ERROR: For MLLM source '{source_name}', designated ID column '{actual_id_col_for_text_loading}' not found. Skipping.")
            return pd.DataFrame()
    elif actual_id_col_for_text_loading not in df.columns: # For human_expert or other sources
        print(f"ERROR: ID column '{actual_id_col_for_text_loading}' not found in '{source_name}' for source '{source_name}'. Skipping.")
        return pd.DataFrame()
    # --- END MODIFICATION ---

    # Check required columns (other than the text loading ID column which is now dynamically determined)
    other_required_cols = []
    if author_col_name_if_direct:
        other_required_cols.append(author_col_name_if_direct)
    if id_col_for_author_extraction:
        other_required_cols.append(id_col_for_author_extraction)
    
    missing_cols = [col for col in other_required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing other required columns in {source_name}: {missing_cols}. Skipping.")
        return pd.DataFrame()
    
    # Extract author labels
    if author_col_name_if_direct:
        print(f"Using direct author column: {author_col_name_if_direct}")
        df['author_label'] = df[author_col_name_if_direct]
    elif id_col_for_author_extraction:
        print(f"Extracting author from: {id_col_for_author_extraction}")
        df['author_label'] = df[id_col_for_author_extraction].apply(
            lambda x: extract_author_name(x.split('/')[0]) if isinstance(x, str) and '/' in x else extract_author_name(x)
        )
    else:
        print("WARNING: No author label extraction method provided.")
        df['author_label'] = "Unknown"
    
    print(f"Unique author labels found: {df['author_label'].nunique()}")
    
    # Load text content
    print(f"Loading text content using column '{actual_id_col_for_text_loading}' from base path: {base_text_path}...")
    df['text_content'] = df[actual_id_col_for_text_loading].apply(
        lambda fid: load_text_from_file_id(
            fid,
            base_text_path,
            append_txt_if_missing=(mllm_append_txt_suffix if source_name == 'mllm' else False)
        )
    )
    
    # Filter out rows where text content could not be loaded
    original_count = len(df)
    df = df.dropna(subset=['text_content'])
    df = df[df['text_content'] != '']  # Also remove empty strings
    if len(df) < original_count:
        print(f"Warning: Dropped {original_count - len(df)} rows due to missing or unloadable text content.")
    
    if df.empty:
        print(f"No text data found for {source_name} after attempting to load from file_ids. Skipping.")
        return pd.DataFrame()
    
    print(f"Successfully loaded text for {len(df)} entries from {source_name}.")
    
    # Add source column
    df['source'] = source_name
    
    # Generate semantic vectors
    if model is not None:
        print(f"Generating semantic vectors for {source_name} data...")
        texts_to_encode = df['text_content'].tolist()
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[i:i+batch_size]
            embeddings_batch = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.extend(embeddings_batch)
            print(f"  Encoded batch {i//batch_size + 1}/{(len(texts_to_encode)-1)//batch_size + 1} for {source_name}")
        
        df_embeddings = pd.DataFrame(all_embeddings)
        print(f"Semantic vectors generated for {source_name}.")
        
        # Prepare output DataFrame
        df_to_save = df[['author_label', 'source', 'text_content']].reset_index(drop=True)
        
        # Add original file ID from the source
        df_to_save['original_file_id'] = df[actual_id_col_for_text_loading].reset_index(drop=True)
        
        # Ensure df_embeddings has a compatible index if original df had a non-standard one before reset_index
        df_embeddings = df_embeddings.reset_index(drop=True)
        
        # Combine metadata with embeddings
        df_vectors_with_author_labels = pd.concat([df_to_save, df_embeddings], axis=1)
        return df_vectors_with_author_labels
    else:
        print("ERROR: No model provided for generating embeddings.")
        return pd.DataFrame()


def generate_author_vectors_and_centroids():
    """
    Generates semantic vectors for both human expert and MLLM commentaries using BAAI/bge-large-zh-v1.5,
    saves them along with author assignments, and calculates author centroids.
    """
    script_dir = Path(__file__).resolve().parent
    
    # Define paths for both data sources
    # human_expert_input_csv_path = script_dir / "../../result/analysis_results/dimensionality_reduction_with_profile_scores.csv" # OLD
    # mllms_input_csv_path = script_dir / "../../../../experiment/MLLMS/result/analysis_results/dimensionality_reduction_with_profile_scores.csv" # OLD
    
    combined_input_csv_path = script_dir / "../../result/analysis_results/dimensionality_reduction_with_profile_scores_human_and_gemini.csv"
    
    # Base paths for text data
    human_expert_base_text_data_path = script_dir / "../../../../data/text/processed/"
    mllms_base_text_data_path = script_dir / "../../../../experiment/MLLMS/feedbacks/gemini2.5/" # Point to gemini2.5 root, specific subfolders like January handled by file_id
    
    # Output paths
    output_dir = script_dir / "../../result/analysis_results"  # Consistent output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_output_csv_path = output_dir / "all_author_semantic_vectors.csv"
    centroids_output_csv_path = output_dir / "all_author_semantic_centroids.csv"

    # print(f"Human expert input CSV: {human_expert_input_csv_path}") # OLD - REMOVE
    # print(f"MLLMs input CSV: {mllms_input_csv_path}") # OLD - REMOVE
    print(f"Combined input CSV: {combined_input_csv_path}") # ADDED/MODIFIED
    print(f"Output vectors will be saved to: {vectors_output_csv_path}")
    print(f"Output centroids will be saved to: {centroids_output_csv_path}")

    # --- Load Sentence Transformer Model ---
    model_name = 'BAAI/bge-large-zh-v1.5'
    print(f"Loading sentence transformer model: {model_name}...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please ensure 'sentence-transformers' and 'torch' are installed.")
        return
    print("Model loaded successfully.")

    # --- Load Combined Data ---
    print(f"Loading combined data from {combined_input_csv_path}...")
    if not combined_input_csv_path.exists():
        print(f"ERROR: Combined input CSV not found at {combined_input_csv_path}.")
        return
    try:
        df_combined = pd.read_csv(combined_input_csv_path)
        print(f"Successfully loaded {len(df_combined)} entries from combined CSV.")
    except Exception as e:
        print(f"Error loading combined CSV: {e}")
        return

    # --- Process Human Expert Data ---
    df_human_data = df_combined[df_combined['source_type'] == 'human_expert'].copy()
    if df_human_data.empty:
        print("Warning: No human expert data found in the combined CSV.")
        df_human_expert = pd.DataFrame()
    else:
        df_human_expert = process_input_source(
            df_input=df_human_data,
            base_text_path=human_expert_base_text_data_path,
            source_name='human_expert',
            id_col_for_text_loading='file_id', # Assuming 'file_id' exists and is correct for human data
            author_col_name_if_direct=None,
            id_col_for_author_extraction='file_id', # Author extracted from 'file_id' for human
            model=model
        )
    
    # --- Process MLLM Data ---
    # Assuming 'source_type' column exists to differentiate
    df_mllm_data = df_combined[df_combined['source_type'] == 'mllm'].copy() 
    if df_mllm_data.empty:
        print("Warning: No MLLM data found in the combined CSV.")
        df_mllm = pd.DataFrame()
    else:
        df_mllm = process_input_source(
            df_input=df_mllm_data,
            base_text_path=mllms_base_text_data_path, # Base path for MLLM texts
            source_name='mllm',
            # Assuming 'file_id' in the combined CSV can be used for MLLM text loading.
            # Or if 'original_filename' for MLLM was carried into combined CSV and is more direct, use that.
            # For consistency, let's assume 'file_id' is populated for MLLM rows too and refers to the text path relative to mllms_base_text_data_path
            id_col_for_text_loading='file_id',  
            author_col_name_if_direct='author_display_name', # If 'author_display_name' from MLLM data is in combined CSV
            id_col_for_author_extraction=None,
            model=model
        )
    
    # --- Combine Data from Both Sources ---
    if df_human_expert.empty and df_mllm.empty:
        print("ERROR: Both data sources failed to process. Exiting.")
        return
    
    if df_human_expert.empty:
        print("Warning: No human expert data processed. Proceeding with MLLM data only.")
        df_all_vectors = df_mllm
    elif df_mllm.empty:
        print("Warning: No MLLM data processed. Proceeding with human expert data only.")
        df_all_vectors = df_human_expert
    else:
        print("Combining data from both sources...")
        df_all_vectors = pd.concat([df_human_expert, df_mllm], ignore_index=True)
    
    print(f"Total combined vectors: {len(df_all_vectors)}")
    
    # --- Save Combined Vectors ---
    print(f"Saving combined vectors to {vectors_output_csv_path}...")
    try:
        df_all_vectors.to_csv(vectors_output_csv_path, index=False, encoding='utf-8-sig')
        print("Combined vectors saved successfully.")
    except Exception as e:
        print(f"Error saving combined vectors CSV: {e}")

    # --- Calculate Author Centroids for Combined Data ---
    print("Calculating author centroids for combined data...")
    vector_dim_start_col = 4  # After 'author_label', 'source', 'text_content', 'original_file_id'
    
    author_centroids = {}
    # Iterate over unique author_label
    for current_author_label in df_all_vectors['author_label'].unique():
        if pd.isna(current_author_label):
            continue
        # Filter vectors for the current author
        author_specific_vectors_df = df_all_vectors[df_all_vectors['author_label'] == current_author_label]
        numeric_vector_data = author_specific_vectors_df.iloc[:, vector_dim_start_col:].values
        
        if numeric_vector_data.shape[0] > 0:
            centroid = np.mean(numeric_vector_data, axis=0)
            author_centroids[current_author_label] = centroid  # Use author label as key
        else:
            print(f"  No vectors found for author: {current_author_label}. Skipping centroid calculation for it.")

    if author_centroids:
        df_centroids = pd.DataFrame.from_dict(author_centroids, orient='index')
        df_centroids.index.name = 'author_label'  # Set index name
        
        # Add a source column that identifies if the author is from 'human_expert', 'mllm', or 'both'
        author_sources = {}
        for author in df_centroids.index:
            human_expert_count = len(df_all_vectors[(df_all_vectors['author_label'] == author) & 
                                                   (df_all_vectors['source'] == 'human_expert')])
            mllm_count = len(df_all_vectors[(df_all_vectors['author_label'] == author) & 
                                           (df_all_vectors['source'] == 'mllm')])
            
            if human_expert_count > 0 and mllm_count > 0:
                author_sources[author] = 'both'
            elif human_expert_count > 0:
                author_sources[author] = 'human_expert'
            else:
                author_sources[author] = 'mllm'
        
        df_centroids['source'] = pd.Series(author_sources)
        
        print(f"Saving author centroids to {centroids_output_csv_path}...")
        try:
            df_centroids.to_csv(centroids_output_csv_path, encoding='utf-8-sig')
            print("Centroids saved successfully.")
        except Exception as e:
            print(f"Error saving centroids CSV: {e}")
    else:
        print("No centroids were calculated.")

    print("Script finished.")

if __name__ == '__main__':
    generate_author_vectors_and_centroids() 