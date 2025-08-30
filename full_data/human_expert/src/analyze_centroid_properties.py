import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
import numpy as np

def calculate_reduced_space_centroids(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Reads dimensionality reduction results, groups by profile assignment (primary_profile_or_status),
    and calculates the mean coordinates (centroids) in t-SNE and UMAP spaces.
    """
    print(f"Calculating reduced space centroids from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    profile_column_for_grouping = 'primary_profile_or_status' # Use the new column name

    # Ensure required columns exist
    # if 'specialized_profile_assignment' not in df.columns or 'tsne_x' not in df.columns or 'tsne_y' not in df.columns: # OLD check
    if profile_column_for_grouping not in df.columns or 'tsne_x' not in df.columns or 'tsne_y' not in df.columns: # NEW check
        print(f"Error: Missing required columns ('{profile_column_for_grouping}', 'tsne_x', 'tsne_y') in input CSV: {input_csv_path}")
        return pd.DataFrame()

    centroids_data = {}
    
    # Calculate t-SNE centroids
    # tsne_centroids = df.groupby('specialized_profile_assignment')[['tsne_x', 'tsne_y']].mean() # OLD grouping
    tsne_centroids = df.groupby(profile_column_for_grouping)[['tsne_x', 'tsne_y']].mean() # NEW grouping
    for profile_name, coords in tsne_centroids.iterrows():
        centroids_data[profile_name] = {'tsne_x': coords['tsne_x'], 'tsne_y': coords['tsne_y']}

    # Calculate UMAP centroids if columns exist
    if 'umap_x' in df.columns and 'umap_y' in df.columns and not df[['umap_x', 'umap_y']].isnull().all().all():
        # umap_centroids = df.groupby('specialized_profile_assignment')[['umap_x', 'umap_y']].mean() # OLD grouping
        umap_centroids = df.groupby(profile_column_for_grouping)[['umap_x', 'umap_y']].mean() # NEW grouping
        for profile_name, coords in umap_centroids.iterrows():
            if profile_name in centroids_data:
                centroids_data[profile_name]['umap_x'] = coords['umap_x']
                centroids_data[profile_name]['umap_y'] = coords['umap_y']
            else: # Should not happen if profile names are consistent
                centroids_data[profile_name] = {'umap_x': coords['umap_x'], 'umap_y': coords['umap_y']}
    else:
        print("UMAP columns ('umap_x', 'umap_y') not found or are all NaN. Skipping UMAP centroid calculation.")

    # Convert to DataFrame
    centroids_df = pd.DataFrame.from_dict(centroids_data, orient='index')
    # Add profile name as a column from index
    # centroids_df.index.name = 'specialized_profile_assignment' # OLD index name
    centroids_df.index.name = profile_column_for_grouping # NEW index name
    centroids_df.reset_index(inplace=True)


    # Fill NaN for UMAP if some profiles had t-SNE but no UMAP data (though groupby().mean() should handle this by not producing rows for them)
    if 'umap_x' not in centroids_df.columns: # if umap was skipped entirely
        centroids_df['umap_x'] = np.nan
        centroids_df['umap_y'] = np.nan
    else: # if umap was processed, ensure all rows have the columns, even if NaN
        # for profile_name in centroids_df['specialized_profile_assignment'].unique(): # OLD iteration
        for profile_name in centroids_df[profile_column_for_grouping].unique(): # NEW iteration
             if 'umap_x' not in centroids_df.columns or centroids_df.loc[centroids_df[profile_column_for_grouping] == profile_name, 'umap_x'].isnull().all():
                  centroids_df.loc[centroids_df[profile_column_for_grouping] == profile_name, 'umap_x'] = np.nan
                  centroids_df.loc[centroids_df[profile_column_for_grouping] == profile_name, 'umap_y'] = np.nan


    # Reorder columns for clarity
    # cols_order = ['specialized_profile_assignment', 'tsne_x', 'tsne_y'] # OLD column order
    cols_order = [profile_column_for_grouping, 'tsne_x', 'tsne_y'] # NEW column order
    if 'umap_x' in centroids_df.columns and 'umap_y' in centroids_df.columns:
        cols_order.extend(['umap_x', 'umap_y'])
    
    # Filter out columns that might not exist if UMAP was skipped before reordering
    existing_cols_order = [col for col in cols_order if col in centroids_df.columns]
    centroids_df = centroids_df[existing_cols_order]

    try:
        centroids_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Reduced space centroids saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving reduced space centroids CSV: {e}")
    return centroids_df

def calculate_high_dim_centroid_similarity(input_centroids_csv_path: str, output_similarity_csv_path: str):
    """
    Reads high-dimensional profile centroids and calculates the cosine similarity matrix between them.
    The input CSV is expected to have 'profile_name' as the first column for labels.
    """
    print(f"Calculating high-dimensional centroid similarity from: {input_centroids_csv_path}")
    try:
        centroids_df = pd.read_csv(input_centroids_csv_path)
    except FileNotFoundError:
        print(f"Error: High-dimensional centroids file not found at {input_centroids_csv_path}")
        return

    # The generate_semantic_vectors.py script saves centroids with the index 'profile_name'.
    # When read_csv loads it, 'profile_name' becomes the first column if no explicit index_col is set.
    profile_label_col = 'profile_name' 
    if profile_label_col not in centroids_df.columns:
        print(f"Error: '{profile_label_col}' column not found in high-dimensional centroids CSV: {input_centroids_csv_path}.")
        print(f"Available columns: {centroids_df.columns.tolist()}")
        # Attempt to use the first column as profile_name if it's non-numeric,
        # and subsequent columns are numeric (vectors)
        if centroids_df.iloc[:, 0].dtype == 'object' and pd.api.types.is_numeric_dtype(centroids_df.iloc[:, 1:].values.flatten()):
            assumed_profile_col = centroids_df.columns[0]
            print(f"Assuming first column '{assumed_profile_col}' is the profile label column.")
            centroids_df.rename(columns={assumed_profile_col: profile_label_col}, inplace=True)
        else:
            return


    profile_names = centroids_df[profile_label_col].tolist()
    # Assuming vector dimensions are all columns except profile_label_col
    vector_columns = [col for col in centroids_df.columns if col != profile_label_col]
    
    if not vector_columns:
        print("Error: No vector columns found in high-dimensional centroids CSV.")
        return
        
    vectors = centroids_df[vector_columns].values

    if vectors.shape[0] < 2:
        print("Error: Need at least two centroids to calculate similarity.")
        return

    similarity_matrix = cosine_similarity(vectors)
    similarity_df = pd.DataFrame(similarity_matrix, index=profile_names, columns=profile_names)

    try:
        similarity_df.to_csv(output_similarity_csv_path, encoding='utf-8-sig')
        print(f"High-dimensional centroid cosine similarity matrix saved to: {output_similarity_csv_path}")
    except Exception as e:
        print(f"Error saving high-dim similarity CSV: {e}")

def calculate_reduced_space_centroid_distances(reduced_centroids_df: pd.DataFrame, space_prefix: str, output_distance_csv_path: str):
    """
    Calculates Euclidean distances between centroids in a reduced space (t-SNE or UMAP).
    Expects reduced_centroids_df to have a column named 'primary_profile_or_status' for labels.
    """
    print(f"Calculating {space_prefix.upper()} centroid distances.")
    if reduced_centroids_df.empty:
        print(f"Reduced centroids DataFrame is empty. Skipping {space_prefix.upper()} distance calculation.")
        return

    profile_column_for_labels = 'primary_profile_or_status' # Use the new column name for labels
    coord_x_col = f'{space_prefix}_x'
    coord_y_col = f'{space_prefix}_y'

    if coord_x_col not in reduced_centroids_df.columns or coord_y_col not in reduced_centroids_df.columns or profile_column_for_labels not in reduced_centroids_df.columns:
        print(f"Error: Required columns ('{profile_column_for_labels}', '{coord_x_col}', '{coord_y_col}') not found in reduced centroids DataFrame.")
        return
    
    # Drop rows where these specific coordinates are NaN before distance calculation
    valid_centroids_df = reduced_centroids_df.dropna(subset=[coord_x_col, coord_y_col])

    if valid_centroids_df.shape[0] < 2:
        print(f"Need at least two valid centroids in {space_prefix.upper()} space to calculate distances. Found {valid_centroids_df.shape[0]}.")
        return

    # profile_names = valid_centroids_df['specialized_profile_assignment'].tolist() # OLD label column
    profile_names = valid_centroids_df[profile_column_for_labels].tolist() # NEW label column
    coordinates = valid_centroids_df[[coord_x_col, coord_y_col]].values

    distance_matrix = euclidean_distances(coordinates)
    distance_df = pd.DataFrame(distance_matrix, index=profile_names, columns=profile_names)

    try:
        distance_df.to_csv(output_distance_csv_path, encoding='utf-8-sig')
        print(f"{space_prefix.upper()} centroid Euclidean distance matrix saved to: {output_distance_csv_path}")
    except Exception as e:
        print(f"Error saving {space_prefix} distances CSV: {e}")

def main():
    script_dir = Path(__file__).resolve().parent # experiment/human_expert/src/
    base_expert_dir = script_dir.parent # experiment/human_expert/
    
    # Input paths
    input_analysis_results_dir_main = base_expert_dir / "result/analysis_results"
    # This is the CSV generated by phase_1_1_benchmark_script_part2_candidate_selection.py, containing primary_profile_or_status
    dimensionality_reduction_csv = input_analysis_results_dir_main / "dimensionality_reduction_with_profile_scores.csv" 
    
    # This is the CSV generated by generate_semantic_vectors.py (after its modification)
    # It contains high-dimensional centroids calculated based on primary_profile_or_status, with 'profile_name' as index/first column.
    high_dim_centroids_csv = input_analysis_results_dir_main / "profile_semantic_centroids.csv" 

    # Output directory for this script's results
    output_analysis_dir = input_analysis_results_dir_main / "centroid_analysis"
    output_analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_analysis_dir}")

    # Output file paths
    reduced_centroids_output_csv = output_analysis_dir / "reduced_space_profile_centroids.csv"
    high_dim_similarity_output_csv = output_analysis_dir / "high_dim_centroid_cosine_similarity.csv"
    tsne_distances_output_csv = output_analysis_dir / "tsne_centroid_euclidean_distances.csv"
    umap_distances_output_csv = output_analysis_dir / "umap_centroid_euclidean_distances.csv"
    summary_md_path = output_analysis_dir / "centroid_analysis_summary.md"

    # --- 1. Calculate Reduced Space Centroids ---
    calculated_reduced_centroids_df = calculate_reduced_space_centroids(
        str(dimensionality_reduction_csv),
        str(reduced_centroids_output_csv)
    )

    # --- 2. Calculate High-Dimensional Centroid Cosine Similarity ---
    calculate_high_dim_centroid_similarity(
        str(high_dim_centroids_csv),
        str(high_dim_similarity_output_csv)
    )

    # --- 3. Calculate Reduced Space Centroid Euclidean Distances ---
    if not calculated_reduced_centroids_df.empty:
        # t-SNE distances
        if 'tsne_x' in calculated_reduced_centroids_df.columns:
            calculate_reduced_space_centroid_distances(
                calculated_reduced_centroids_df,
                "tsne",
                str(tsne_distances_output_csv)
            )
        else:
            print("Skipping t-SNE distance calculation as 'tsne_x' column is missing in reduced centroids.")

        # UMAP distances (if UMAP data exists and was processed)
        if 'umap_x' in calculated_reduced_centroids_df.columns and not calculated_reduced_centroids_df['umap_x'].isnull().all():
            calculate_reduced_space_centroid_distances(
                calculated_reduced_centroids_df,
                "umap",
                str(umap_distances_output_csv)
            )
        else:
            print("Skipping UMAP distance calculation as 'umap_x' column is missing or all NaN in reduced centroids.")
    else:
        print("Skipping reduced space distance calculations as reduced_centroids_df is empty.")

    # --- 4. Generate Markdown Summary ---
    profile_status_col_name = 'primary_profile_or_status' # For consistency in the report
    md_content = f"""# Centroid Analysis Summary

This document summarizes the outputs of the `analyze_centroid_properties.py` script.

## 1. Reduced Space Profile Centroids

- **File:** `{reduced_centroids_output_csv.name}`
- **Location:** `{reduced_centroids_output_csv.parent}`
- **Description:** This file contains the calculated centroids for each '{profile_status_col_name}' in the t-SNE and UMAP (if available) reduced-dimensional spaces. Centroids are calculated as the mean of 'tsne_x'/'tsne_y' and 'umap_x'/'umap_y' coordinates for all text snippets belonging to a specific profile status.

## 2. High-Dimensional Centroid Cosine Similarity

- **File:** `{high_dim_similarity_output_csv.name}`
- **Location:** `{high_dim_similarity_output_csv.parent}`
- **Description:** This file contains a matrix showing the cosine similarity between the high-dimensional semantic centroids of different profile statuses (derived from '{profile_status_col_name}'). These high-dimensional centroids were originally calculated by `generate_semantic_vectors.py` (after its modification to use '{profile_status_col_name}') and stored in `profile_semantic_centroids.csv` (where labels are under 'profile_name' column). Values range from -1 to 1, where 1 indicates identical vectors (highest similarity).

## 3. t-SNE Space Centroid Euclidean Distances

- **File:** `{tsne_distances_output_csv.name}`
- **Location:** `{tsne_distances_output_csv.parent}`
- **Description:** This file contains a matrix showing the Euclidean distances between the calculated 2D centroids of different profile statuses in the t-SNE space (derived from `reduced_space_profile_centroids.csv`, using the '{profile_status_col_name}' for labeling). Lower values indicate closer proximity in the t-SNE visualization.

"""
    # Check if UMAP results are valid before adding the section to the report
    umap_section_added = False
    if not calculated_reduced_centroids_df.empty and \
       'umap_x' in calculated_reduced_centroids_df.columns and \
       not calculated_reduced_centroids_df['umap_x'].isnull().all() and \
       (output_analysis_dir / umap_distances_output_csv.name).exists():
        
        # Correctly format the multi-line f-string for the UMAP section
        umap_section = f"""## 4. UMAP Space Centroid Euclidean Distances

- **File:** `{umap_distances_output_csv.name}`
- **Location:** `{umap_distances_output_csv.parent}`
- **Description:** This file contains a matrix showing the Euclidean distances between the calculated 2D centroids of different profile statuses in the UMAP space (derived from `reduced_space_profile_centroids.csv`, using '{profile_status_col_name}' for labeling). This is only generated if UMAP coordinates were available and processed. Lower values indicate closer proximity in the UMAP visualization.
"""
        md_content += umap_section # Append the correctly formatted section
        umap_section_added = True
        
    # Correct indentation for the final part
    if not umap_section_added:
        md_content += "\n*UMAP distance analysis was skipped as UMAP data was not available, not processed, or the reduced centroids data was empty.*\n"

    try:
        with open(summary_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown summary saved to: {summary_md_path}")
    except Exception as e:
        print(f"Error saving Markdown summary: {e}")

    print("\nScript execution finished.")

if __name__ == '__main__':
    main() 