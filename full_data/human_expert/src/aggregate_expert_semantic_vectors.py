import pandas as pd
import re
from pathlib import Path
import numpy as np

# --- file_id 解析函数 (与 aggregate_expert_features.py 中的版本保持一致) ---
def extract_scholar_from_file_id(file_id_str: str):
    if not isinstance(file_id_str, str):
        return "Unknown_Scholar"
    part_before_separator = file_id_str.split('\\')[0].split('/')[0]
    match_with_en = re.match(r"([^（]+（[^）]+）)", part_before_separator)
    if match_with_en:
        return match_with_en.group(1)
    if '_' in part_before_separator:
        return part_before_separator.split('_', 1)[0]
    return part_before_separator

def extract_book_id_from_file_id(file_id_str: str):
    if not isinstance(file_id_str, str):
        return "Unknown_Book"
    return file_id_str.split('\\')[0].split('/')[0]

# --- 语义向量聚合函数 ---
def aggregate_vectors(df: pd.DataFrame, group_by_column: str, vector_cols: list) -> pd.DataFrame:
    print(f"Aggregating semantic vectors grouped by: {group_by_column}...")
    if df.empty or group_by_column not in df.columns or not vector_cols or not all(col in df.columns for col in vector_cols):
        print(f"DataFrame is empty, group_by_column '{group_by_column}' is missing, or vector_cols are invalid/missing. Skipping aggregation.")
        return pd.DataFrame()

    # Ensure vector columns are numeric, coerce errors to NaN
    for col in vector_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where all vector components are NaN after coercion, as they can't contribute to mean
    # df_cleaned_vectors = df.dropna(subset=vector_cols, how='all') 
    # Re-evaluating: groupby().mean() naturally handles NaNs by excluding them from mean calculation.
    # If a group has all NaNs for a vector dim, result will be NaN. If a row has all NaNs, it won't contribute to any mean.
    # This is generally fine. The main concern is if entire groups have no valid vectors.

    aggregated_df = df.groupby(group_by_column)[vector_cols].mean().reset_index()
    
    # Check for groups that might have resulted in all NaNs if all their vectors were NaN for all dimensions
    # This check is somewhat redundant if groupby().mean() produces rows with NaNs, 
    # but it's a safeguard if we want to explicitly remove such groups.
    # aggregated_df = aggregated_df.dropna(subset=vector_cols, how='all')
    
    return aggregated_df

def main():
    script_dir = Path(__file__).resolve().parent # experiment/human_expert/src/
    # base_expert_dir = script_dir.parent # experiment/human_expert/

    # Corrected input path based on file search results
    input_vectors_csv_path = script_dir / "result/analysis_results/human_expert_semantic_vectors_with_profiles.csv"
    
    # Output paths
    # Output directory will be experiment/human_expert/result/analysis_results/
    # This is because base_expert_dir points to experiment/human_expert/
    # and output_dir is base_expert_dir / "result/analysis_results"
    # Let's ensure output_dir is correctly defined relative to script_dir.parent for consistency
    base_expert_dir_for_output = script_dir.parent # experiment/human_expert/
    output_dir = base_expert_dir_for_output / "result/analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_scholar_centroids_csv_path = output_dir / "aggregated_scholar_semantic_centroids.csv"
    output_book_centroids_csv_path = output_dir / "aggregated_book_semantic_centroids.csv"

    print(f"Loading semantic vectors from: {input_vectors_csv_path}")
    if not input_vectors_csv_path.exists():
        print(f"ERROR: Input semantic vectors CSV file not found at {input_vectors_csv_path}")
        print("Please ensure the path is correct and the file from phase5 (generate_semantic_vectors.py) exists.")
        return
    df_vectors = pd.read_csv(input_vectors_csv_path)
    print(f"Successfully loaded {len(df_vectors)} rows with semantic vectors.")

    # 假设 'file_id' 列存在于输入的向量CSV中
    if 'file_id' not in df_vectors.columns:
        # Fallback: Try to find a column that looks like file_id if 'file_id' is not present
        # This is a simple heuristic and might need adjustment based on actual CSV structure
        potential_id_cols = [col for col in df_vectors.columns if 'id' in col.lower() or 'file' in col.lower()]
        if potential_id_cols:
            # A more robust check would be to see if this column's values match the expected file_id format
            # For now, let's assume the first one found is the file_id column.
            # If generate_semantic_vectors.py produced it, it *should* be 'file_id' or its equivalent.
            # The `human_expert_semantic_vectors_with_profiles.csv` *does* have `file_id`.
            assumed_file_id_col = 'file_id' # Hardcoding based on known output of generate_semantic_vectors.py
            if assumed_file_id_col not in df_vectors.columns and potential_id_cols:
                 assumed_file_id_col = potential_id_cols[0]
                 print(f"Warning: 'file_id' column not found. Assuming '{assumed_file_id_col}' is the identifier column.")
                 df_vectors.rename(columns={assumed_file_id_col: 'file_id'}, inplace=True)
            elif assumed_file_id_col not in df_vectors.columns: # if still not found after trying potential_id_cols
                 print(f"ERROR: 'file_id' column not found, and no suitable alternative ID column detected in {input_vectors_csv_path}.")
                 return       
        else:
            print(f"ERROR: 'file_id' column not found in {input_vectors_csv_path}. Cannot proceed with ID extraction.")
            return
    
    print("Extracting scholar_id and book_id from file_id...")
    df_vectors['scholar_id'] = df_vectors['file_id'].apply(extract_scholar_from_file_id)
    df_vectors['book_id'] = df_vectors['file_id'].apply(extract_book_id_from_file_id)
    print(df_vectors[['file_id', 'scholar_id', 'book_id']].head())

    # 识别向量列 (例如 '0' 到 '1023' for BAAI/bge-large-zh-v1.5 which is 1024-dim)
    # A more robust way would be to infer this if not strictly named '0'-'1023'
    # but for now, this is standard for sentence-transformer outputs when converted to df columns.
    expected_dim = 1024 # For BAAI/bge-large-zh-v1.5
    vector_cols = [str(i) for i in range(expected_dim)]
    
    # Check if all expected vector columns exist
    missing_vector_cols = [col for col in vector_cols if col not in df_vectors.columns]
    if missing_vector_cols:
        print(f"ERROR: Missing expected vector dimension columns in the input CSV. Example missing: {missing_vector_cols[:5]}")
        print(f"Please ensure '{input_vectors_csv_path}' contains columns named '0' through '{expected_dim-1}'.")
        return
    print(f"Identified {len(vector_cols)} vector dimension columns.")

    # --- Aggregate by Scholar ---
    scholar_centroids_df = aggregate_vectors(df_vectors, "scholar_id", vector_cols)
    if not scholar_centroids_df.empty:
        scholar_centroids_df.to_csv(output_scholar_centroids_csv_path, index=False, encoding='utf-8-sig')
        print(f"Aggregated scholar semantic centroids saved to: {output_scholar_centroids_csv_path}")
    else:
        print("Scholar semantic vector aggregation resulted in an empty DataFrame. No output file created.")


    # --- Aggregate by Book ---
    book_centroids_df = aggregate_vectors(df_vectors, "book_id", vector_cols)
    if not book_centroids_df.empty:
        book_centroids_df.to_csv(output_book_centroids_csv_path, index=False, encoding='utf-8-sig')
        print(f"Aggregated book semantic centroids saved to: {output_book_centroids_csv_path}")
    else:
        print("Book semantic vector aggregation resulted in an empty DataFrame. No output file created.")

    print("Script aggregate_expert_semantic_vectors.py finished.")

if __name__ == '__main__':
    main() 