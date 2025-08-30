import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import torch


def load_text_from_file_id(file_id_str: str, base_data_path: Path):
    """Loads text content from a file path constructed from file_id_str and base_data_path."""
    if not file_id_str or pd.isna(file_id_str):
        # print(f"Warning: Received empty or NaN file_id. Skipping text loading.")
        return None
    try:
        # Assuming file_id_str is like "FolderName/actual_file_name.txt"
        # or just "actual_file_name.txt" if it's directly in a known subfolder not captured by file_id_str structure.
        # For this project, file_id structure is typically FolderName/filename.txt
        # e.g., 乔迅（Jonathan Hay）_清初中国的绘画与现代性/1.1_石涛画像.txt
        full_text_path = base_data_path / file_id_str
        # print(f"[DEBUG] Attempting to load file_id: '{file_id_str}' from path: {full_text_path}") # DEBUG removed for cleaner output
        
        if not full_text_path.exists():
            print(f"[DEBUG] File NOT FOUND for file_id: '{file_id_str}' at resolved path: {full_text_path.resolve()}")
            return None
        
        with open(full_text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading text for file_id '{file_id_str}': {e} at path {full_text_path}")
        return None


def process_input_source(df_input: pd.DataFrame, base_text_path: Path, source_name: str, id_col_for_text_loading: str,
                         profile_col_name: str, model=None):
    """
    Process a single input source DataFrame, extracting text, profile status, and generating embeddings.
    
    Parameters:
    - df_input: Input DataFrame
    - base_text_path: Base directory for loading text files
    - source_name: Name of the source ('human_expert' or 'mllm')
    - id_col_for_text_loading: Column name containing the identifier for loading text files
    - profile_col_name: Column containing profile status information
    - model: SentenceTransformer model for generating embeddings
    
    Returns:
    - DataFrame with processed data including embeddings
    """
    print(f"\nProcessing {source_name} data from DataFrame (shape: {df_input.shape})...")

    df = df_input.copy() # Work on a copy

    if df.empty:
        print(f"Input DataFrame for {source_name} is empty. Skipping.")
        return pd.DataFrame()

    # Check required columns
    required_columns = [id_col_for_text_loading, profile_col_name]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns in {source_name} DataFrame: {missing_cols}")
        return pd.DataFrame()
    
    print(f"Successfully loaded {len(df)} entries with file_ids and profile status from {source_name} DataFrame.")
    
    # Load text content
    print(f"Loading text content from base path: {base_text_path}...")
    df['text_content'] = df[id_col_for_text_loading].apply(
        lambda fid: load_text_from_file_id(fid, base_text_path)
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
        df_to_save = df[[profile_col_name, 'source', 'text_content']].reset_index(drop=True)
        
        # Add original file ID from the source
        df_to_save['original_file_id'] = df[id_col_for_text_loading].reset_index(drop=True)
        
        # Ensure df_embeddings has a compatible index if original df had a non-standard one before reset_index
        df_embeddings = df_embeddings.reset_index(drop=True)
        
        # Combine metadata with embeddings
        df_vectors_with_profiles = pd.concat([df_to_save, df_embeddings], axis=1)
        return df_vectors_with_profiles
    else:
        print("ERROR: No model provided for generating embeddings.")
        return pd.DataFrame()


def generate_vectors_and_centroids():
    """
    Generates semantic vectors for both human expert and MLLM commentaries using BAAI/bge-large-zh-v1.5,
    saves them along with profile assignments, and calculates profile centroids based on primary_profile_or_status.
    """
    script_dir = Path(__file__).resolve().parent
    
    # Define paths for both data sources
    # human_expert_input_csv_path = script_dir / "../../result/analysis_results/dimensionality_reduction_with_profile_scores.csv" # OLD
    # mllms_input_csv_path = script_dir / "../../../../experiment/MLLMS/result/analysis_results/dimensionality_reduction_with_profile_scores.csv" # OLD

    combined_input_csv_path = script_dir / "../../result/analysis_results/dimensionality_reduction_with_profile_scores_human_and_gemini.csv"

    # Base paths for text data
    human_expert_base_text_data_path = script_dir / "../../../../data/text/processed/"
    mllms_base_text_data_path = script_dir / "../../../../experiment/MLLMS/feedbacks/gemini2.5/" # Point to gemini2.5 root
    
    # Output paths
    output_dir = script_dir / "../../result/analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Update output paths to reflect combined data
    vectors_output_csv_path = output_dir / "all_semantic_vectors_with_profiles.csv"
    centroids_output_csv_path = output_dir / "all_profile_semantic_centroids.csv"

    # Profile column name is the same for both sources
    profile_column_name = 'primary_profile_or_status'

    print(f"Combined input CSV: {combined_input_csv_path}")
    print(f"Profile column name for both sources: {profile_column_name}")
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
            id_col_for_text_loading='file_id', # Assuming 'file_id' is correct for human data
            profile_col_name=profile_column_name,
            model=model
        )
    
    # --- Process MLLM Data ---
    df_mllm_data = df_combined[df_combined['source_type'] == 'mllm'].copy()
    if df_mllm_data.empty:
        print("Warning: No MLLM data found in the combined CSV.")
        df_mllm = pd.DataFrame()
    else:
        df_mllm = process_input_source(
            df_input=df_mllm_data,
            base_text_path=mllms_base_text_data_path,
            source_name='mllm',
            # Assuming 'file_id' is populated for MLLM rows in the combined CSV
            id_col_for_text_loading='file_id', 
            profile_col_name=profile_column_name,
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

    # --- Calculate Profile Centroids for Combined Data ---
    print(f"Calculating centroids based on '{profile_column_name}'...")
    vector_dim_start_col = 4  # After profile_column_name, 'source', 'text_content', 'original_file_id'
    
    profile_centroids = {}
    # Iterate over unique values in the profile_column_name
    for profile_name in df_all_vectors[profile_column_name].unique():
        if pd.isna(profile_name):
            print("  Skipping NaN profile name.")
            continue
            
        # Filter vectors for the current profile status
        profile_specific_vectors_df = df_all_vectors[df_all_vectors[profile_column_name] == profile_name]
        numeric_vector_data = profile_specific_vectors_df.iloc[:, vector_dim_start_col:].values
        
        if numeric_vector_data.shape[0] > 0:
            centroid = np.mean(numeric_vector_data, axis=0)
            profile_centroids[profile_name] = centroid
        else:
            print(f"  No vectors found for profile status: {profile_name}. Skipping centroid calculation for it.")

    if profile_centroids:
        df_centroids = pd.DataFrame.from_dict(profile_centroids, orient='index')
        df_centroids.index.name = 'profile_name'
        
        # Add a source column that identifies if the profile is from 'human_expert', 'mllm', or 'both'
        profile_sources = {}
        for profile in df_centroids.index:
            human_expert_count = len(df_all_vectors[(df_all_vectors[profile_column_name] == profile) & 
                                                   (df_all_vectors['source'] == 'human_expert')])
            mllm_count = len(df_all_vectors[(df_all_vectors[profile_column_name] == profile) & 
                                           (df_all_vectors['source'] == 'mllm')])
            
            if human_expert_count > 0 and mllm_count > 0:
                profile_sources[profile] = 'both'
            elif human_expert_count > 0:
                profile_sources[profile] = 'human_expert'
            else:
                profile_sources[profile] = 'mllm'
        
        df_centroids['source'] = pd.Series(profile_sources)
        
        print(f"Saving profile centroids to {centroids_output_csv_path}...")
        try:
            df_centroids.to_csv(centroids_output_csv_path, encoding='utf-8-sig')
            print("Centroids saved successfully.")
        except Exception as e:
            print(f"Error saving centroids CSV: {e}")
    else:
        print("No centroids were calculated.")

    print("Script finished.")

if __name__ == '__main__':
    generate_vectors_and_centroids() 