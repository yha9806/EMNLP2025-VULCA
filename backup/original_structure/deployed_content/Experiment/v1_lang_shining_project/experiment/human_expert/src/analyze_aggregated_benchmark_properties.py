import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
import umap # umap-learn package
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# --- Configure Matplotlib for Chinese font display ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Source Han Sans CN', 'Noto Sans CJK SC'] # Add more as needed
    plt.rcParams['axes.unicode_minus'] = False  # Correctly display negative signs
    print("Attempted to set Matplotlib font for Chinese characters (e.g., SimHei, Microsoft YaHei).")
except Exception as e:
    print(f"Could not set Matplotlib font parameters: {e}")
    print("Please ensure you have a suitable Chinese font installed and Matplotlib can find it.")
    print("Common fonts: SimHei, Microsoft YaHei, WenQuanYi Zen Hei, Source Han Sans CN")

# --- Global Configuration ---
BASE_OUTPUT_DIR = Path("experiment/human_expert/result/analysis_results/aggregated_benchmark_analysis/")
SCHOLAR_FEATURE_DIST_DIR = BASE_OUTPUT_DIR / "scholar_feature_distributions"
BOOK_FEATURE_DIST_DIR = BASE_OUTPUT_DIR / "book_feature_distributions"

# --- Helper Functions ---
def ensure_dir(directory_path: Path):
    """Ensures that a directory exists, creating it if necessary."""
    directory_path.mkdir(parents=True, exist_ok=True)

# --- Core Analysis Functions ---

def load_data(scholar_features_path: Path, book_features_path: Path,
              scholar_centroids_path: Path, book_centroids_path: Path) -> tuple:
    """Loads all necessary aggregated data files."""
    print(f"Loading scholar features from: {scholar_features_path}")
    df_scholar_features = pd.read_csv(scholar_features_path)
    
    print(f"Loading book features from: {book_features_path}")
    df_book_features = pd.read_csv(book_features_path)
    
    print(f"Loading scholar centroids from: {scholar_centroids_path}")
    # Assuming the first column is the ID (scholar_id)
    df_scholar_centroids = pd.read_csv(scholar_centroids_path).set_index('scholar_id') 
    
    print(f"Loading book centroids from: {book_centroids_path}")
    # Assuming the first column is the ID (book_id)
    df_book_centroids = pd.read_csv(book_centroids_path).set_index('book_id')
    
    print("Data loading complete.")
    return df_scholar_features, df_book_features, df_scholar_centroids, df_book_centroids

def analyze_aggregated_features(df_features: pd.DataFrame, aggregation_level_name: str, output_dir: Path, feature_dist_dir: Path):
    """Analyzes aggregated features (descriptive stats, visualizations)."""
    print(f"\\n--- Analyzing aggregated features for: {aggregation_level_name} ---")
    ensure_dir(feature_dist_dir)

    # 1. Descriptive Statistics
    print("Calculating descriptive statistics...")
    # Identify numerical columns (heuristics: 'avg_', 'median_', 'num_segments', 'stance_score')
    numerical_cols = [col for col in df_features.columns if 
                      col.startswith('avg_') or 
                      col.startswith('median_') or 
                      col == 'num_segments' or 
                      col.endswith('_score')]
    
    if not numerical_cols:
        print(f"Warning: No numerical columns found for descriptive statistics for {aggregation_level_name}.")
        desc_stats_df = pd.DataFrame()
    else:
        desc_stats_df = df_features[numerical_cols].describe()
    
    desc_stats_path = output_dir / f"{aggregation_level_name}_features_descriptive_stats.csv"
    desc_stats_df.to_csv(desc_stats_path)
    print(f"Descriptive statistics saved to: {desc_stats_path}")

    # Categorical feature: Stance Label
    stance_col_zh = 'mode_stance_label_zh'
    if stance_col_zh in df_features.columns:
        print(f"Analyzing stance distribution ({stance_col_zh})...")
        stance_counts = df_features[stance_col_zh].value_counts()
        print(f"Stance counts for {aggregation_level_name}:\\n{stance_counts}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=stance_counts.index, y=stance_counts.values)
        plt.title(f'Stance Distribution for {aggregation_level_name.capitalize()}')
        plt.xlabel('Stance Label (Chinese)')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        stance_dist_path = feature_dist_dir / f"{aggregation_level_name}_stance_distribution.png"
        plt.savefig(stance_dist_path)
        plt.close()
        print(f"Stance distribution plot saved to: {stance_dist_path}")
    else:
        print(f"Warning: Column '{stance_col_zh}' not found in {aggregation_level_name} features.")

    # Example: Visualize one numerical feature (e.g., first 'avg_feat_..._score' column found)
    avg_feature_score_cols = [col for col in numerical_cols if col.startswith('avg_feat_')]
    if avg_feature_score_cols:
        example_feature_col = avg_feature_score_cols[0] # Take the first one
        print(f"Visualizing distribution for example feature: {example_feature_col}")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df_features[example_feature_col].dropna(), kde=True)
        plt.title(f'Distribution of {example_feature_col} for {aggregation_level_name.capitalize()}')
        plt.xlabel(example_feature_col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        feature_hist_path = feature_dist_dir / f"{aggregation_level_name}_{example_feature_col}_distribution.png"
        plt.savefig(feature_hist_path)
        plt.close()
        print(f"Example feature distribution plot saved to: {feature_hist_path}")
    else:
        print(f"Warning: No 'avg_feat_..._score' columns found for example distribution plot for {aggregation_level_name}.")

def analyze_aggregated_semantic_centroids(df_centroids: pd.DataFrame, aggregation_level_name: str, output_dir: Path):
    """Analyzes aggregated semantic centroids (similarity, dimensionality reduction, visualization)."""
    print(f"\\n--- Analyzing aggregated semantic centroids for: {aggregation_level_name} ---")
    if df_centroids.empty:
        print(f"Warning: Centroids DataFrame for {aggregation_level_name} is empty. Skipping analysis.")
        return

    # Identify vector columns (assuming they are all columns except the ID, which is the index)
    # Or more robustly, assuming they are string-named integers '0', '1', ...
    vector_cols = [col for col in df_centroids.columns if col.isdigit()]
    if not vector_cols: # Fallback if columns are not named '0', '1', ...
        vector_cols = df_centroids.columns.tolist()
        print(f"Warning: Could not identify vector columns by digit names. Assuming all columns are vector components for {aggregation_level_name}.")

    if not vector_cols or df_centroids[vector_cols].empty:
         print(f"Warning: No valid vector columns found or data is empty for {aggregation_level_name}. Skipping semantic analysis.")
         return
         
    semantic_vectors = df_centroids[vector_cols].copy()
    # Ensure vectors are numeric and handle potential NaNs by filling with mean of the column
    # This is a simple imputation strategy; more sophisticated ones could be used.
    for col in semantic_vectors.columns:
        semantic_vectors[col] = pd.to_numeric(semantic_vectors[col], errors='coerce')
    semantic_vectors = semantic_vectors.fillna(semantic_vectors.mean())


    # 1. High-dimensional Cosine Similarity
    print("Calculating high-dimensional cosine similarity...")
    cosine_sim_matrix = cosine_similarity(semantic_vectors)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df_centroids.index, columns=df_centroids.index)
    cosine_sim_path = output_dir / f"{aggregation_level_name}_high_dim_centroid_cosine_similarity.csv"
    cosine_sim_df.to_csv(cosine_sim_path)
    print(f"High-dimensional cosine similarity matrix saved to: {cosine_sim_path}")

    num_samples = len(df_centroids)
    if num_samples <= 1:
        print(f"Warning: Not enough samples ({num_samples}) for {aggregation_level_name} to perform dimensionality reduction. Skipping.")
        # Create empty files for consistency if needed by downstream processes
        empty_df_cols = ['id', 'tsne_x', 'tsne_y', 'umap_x', 'umap_y'] if aggregation_level_name == "scholar" else ['id', 'tsne_x', 'tsne_y', 'umap_x', 'umap_y'] # Adjust 'id' based on index name
        empty_reduced_df = pd.DataFrame(columns=empty_df_cols) #TODO: fix id column name
        empty_reduced_df.to_csv(output_dir / f"{aggregation_level_name}_reduced_space_centroids.csv", index=False)
        
        placeholder_dist_df = pd.DataFrame(index=df_centroids.index, columns=df_centroids.index)
        placeholder_dist_df.to_csv(output_dir / f"{aggregation_level_name}_tsne_centroid_distances.csv")
        placeholder_dist_df.to_csv(output_dir / f"{aggregation_level_name}_umap_centroid_distances.csv")
        # Create empty placeholder images
        plt.figure() 
        plt.title(f"Not enough data for {aggregation_level_name} t-SNE")
        plt.savefig(output_dir / f"{aggregation_level_name}_semantic_space_tsne.png")
        plt.close()
        plt.figure()
        plt.title(f"Not enough data for {aggregation_level_name} UMAP")
        plt.savefig(output_dir / f"{aggregation_level_name}_semantic_space_umap.png")
        plt.close()
        return


    # 2. Dimensionality Reduction (t-SNE and UMAP)
    print("Performing dimensionality reduction (t-SNE and UMAP)...")
    perplexity_val = min(30, num_samples - 1) if num_samples > 1 else 5 # Ensure perplexity < n_samples
    n_neighbors_val = min(15, num_samples - 1) if num_samples > 1 else 5 # Ensure n_neighbors < n_samples
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300, learning_rate='auto' if hasattr(TSNE, '_check_params_vs_input') else 200.0) # Auto LR for newer sklearn
    tsne_results = tsne.fit_transform(semantic_vectors)
    
    # Ensure UMAP can run, n_neighbors must be less than the number of samples
    if num_samples <= n_neighbors_val: #This check might be redundant given n_neighbors_val calculation, but good for safety
        print(f"Warning: Cannot run UMAP for {aggregation_level_name} as n_samples ({num_samples}) <= n_neighbors ({n_neighbors_val}). Using t-SNE results for UMAP placeholders if possible.")
        umap_results = np.full((num_samples, 2), np.nan) # Placeholder with NaNs
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_val, min_dist=0.1)
        umap_results = reducer.fit_transform(semantic_vectors)

    reduced_df = pd.DataFrame({
        'id': df_centroids.index, # or .reset_index()[df_centroids.index.name]
        'tsne_x': tsne_results[:, 0],
        'tsne_y': tsne_results[:, 1],
        'umap_x': umap_results[:, 0],
        'umap_y': umap_results[:, 1]
    })
    reduced_path = output_dir / f"{aggregation_level_name}_reduced_space_centroids.csv"
    reduced_df.to_csv(reduced_path, index=False)
    print(f"Reduced space centroids (t-SNE & UMAP) saved to: {reduced_path}")

    # 3. Visualization in Reduced Space
    print("Visualizing reduced space (t-SNE and UMAP)...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='tsne_x', y='tsne_y', data=reduced_df, hue='id', legend='brief', s=100)
    plt.title(f't-SNE visualization of {aggregation_level_name} Semantic Centroids')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # For brief legend, or remove if too many items
    if len(df_centroids.index) > 20 : plt.legend().set_visible(False) 
    plt.tight_layout()
    tsne_plot_path = output_dir / f"{aggregation_level_name}_semantic_space_tsne.png"
    plt.savefig(tsne_plot_path)
    plt.close()
    print(f"t-SNE visualization saved to: {tsne_plot_path}")

    if not np.isnan(umap_results).all(): # Only plot if UMAP results are not all NaNs
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='umap_x', y='umap_y', data=reduced_df, hue='id', legend='brief', s=100)
        plt.title(f'UMAP visualization of {aggregation_level_name} Semantic Centroids')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        if len(df_centroids.index) > 20 : plt.legend().set_visible(False)
        plt.tight_layout()
        umap_plot_path = output_dir / f"{aggregation_level_name}_semantic_space_umap.png"
        plt.savefig(umap_plot_path)
        plt.close()
        print(f"UMAP visualization saved to: {umap_plot_path}")
    else:
        print(f"Skipping UMAP visualization for {aggregation_level_name} due to UMAP execution issues.")


    # 4. Euclidean Distances in Reduced Space
    print("Calculating Euclidean distances in reduced space...")
    tsne_coords = reduced_df[['tsne_x', 'tsne_y']].values
    tsne_distances = euclidean_distances(tsne_coords)
    tsne_dist_df = pd.DataFrame(tsne_distances, index=df_centroids.index, columns=df_centroids.index)
    tsne_dist_path = output_dir / f"{aggregation_level_name}_tsne_centroid_distances.csv"
    tsne_dist_df.to_csv(tsne_dist_path)
    print(f"t-SNE centroid distances saved to: {tsne_dist_path}")

    if not np.isnan(umap_results).all():
        umap_coords = reduced_df[['umap_x', 'umap_y']].dropna().values # Drop NaN rows if UMAP failed for some
        if umap_coords.shape[0] > 1: # Check if there are enough points for distance calculation
            umap_distances = euclidean_distances(umap_coords)
            # Need to align indices correctly if some rows were dropped
            valid_indices_for_umap = reduced_df.dropna(subset=['umap_x', 'umap_y']).index
            umap_dist_df = pd.DataFrame(umap_distances, index=reduced_df.loc[valid_indices_for_umap, 'id'], columns=reduced_df.loc[valid_indices_for_umap, 'id'])
            umap_dist_path = output_dir / f"{aggregation_level_name}_umap_centroid_distances.csv"
            umap_dist_df.to_csv(umap_dist_path)
            print(f"UMAP centroid distances saved to: {umap_dist_path}")
        else:
            print(f"Not enough valid UMAP coordinates to calculate distances for {aggregation_level_name}. Skipping UMAP distance matrix.")
            # Create placeholder
            pd.DataFrame(index=df_centroids.index, columns=df_centroids.index).to_csv(output_dir / f"{aggregation_level_name}_umap_centroid_distances.csv")
    else:
        print(f"Skipping UMAP distance calculation for {aggregation_level_name} due to UMAP execution issues.")
        pd.DataFrame(index=df_centroids.index, columns=df_centroids.index).to_csv(output_dir / f"{aggregation_level_name}_umap_centroid_distances.csv")


def generate_summary_report_placeholder(analysis_output_dir: Path, aggregation_level_name: str):
    """Generates a placeholder Markdown summary report."""
    print(f"\\nGenerating placeholder summary report for: {aggregation_level_name}...")
    report_path = analysis_output_dir / f"aggregated_{aggregation_level_name}_benchmark_analysis_summary.md"
    
    content = f"""# Analysis Summary for Aggregated {aggregation_level_name.capitalize()} Benchmarks

This report summarizes the analysis of aggregated human expert benchmarks at the **{aggregation_level_name}** level.
Please review the generated CSV files and plots to draw conclusions about benchmark distinctiveness.

## 1. Aggregated Feature Analysis

### Descriptive Statistics
- **Path:** `{aggregation_level_name}_features_descriptive_stats.csv`
- *Review this file for mean, median, std dev, etc., of key aggregated features.*

### Feature Distributions
- **Stance Distribution Plot:** `feature_distributions/{aggregation_level_name}_stance_distribution.png` (inside `{aggregation_level_name}_feature_distributions` subdirectory relative to this report location if this markdown is moved to `aggregated_benchmark_analysis/`)
- **Example Feature Distribution Plot:** (e.g., `feature_distributions/{aggregation_level_name}_[feature_name]_distribution.png`)
- *Observe the spread and central tendencies of features like stance, average scores for specific 'features' or 'quality' dimensions.*

**Key Observations & Discussion Points for Feature Analysis:**
- *[TODO: Add your observations about feature distinctiveness based on stats and plots. Are there clear differences between {aggregation_level_name}s?]*
- *[TODO: How do these aggregated features compare to individual segment features or profile-based features?]*


## 2. Aggregated Semantic Centroid Analysis

### High-Dimensional Similarity
- **Cosine Similarity Matrix:** `{aggregation_level_name}_high_dim_centroid_cosine_similarity.csv`
- *Examine this matrix. Are the centroids generally distinct (low similarity) or similar (high similarity) in the original high-dimensional semantic space?*

### Dimensionality Reduction & Visualization
- **Reduced Space Coordinates (t-SNE & UMAP):** `{aggregation_level_name}_reduced_space_centroids.csv`
- **t-SNE Visualization:** `{aggregation_level_name}_semantic_space_tsne.png`
- **UMAP Visualization:** `{aggregation_level_name}_semantic_space_umap.png`
- *Do the visualizations show clear clusters for different {aggregation_level_name}s? Or are they largely overlapping?*

### Distances in Reduced Space
- **t-SNE Euclidean Distances:** `{aggregation_level_name}_tsne_centroid_distances.csv`
- **UMAP Euclidean Distances:** `{aggregation_level_name}_umap_centroid_distances.csv`
- *Quantitatively, how far apart are the centroids in the 2D spaces?*

**Key Observations & Discussion Points for Semantic Centroid Analysis:**
- *[TODO: Discuss the distinctiveness of semantic centroids. Does aggregation (by {aggregation_level_name}) lead to more separable representations compared to individual segments or original profiles?]*
- *[TODO: Compare t-SNE and UMAP results. Do they provide consistent insights?]*

## Overall Conclusion
- *[TODO: Summarize the findings. Are the aggregated benchmarks (by {aggregation_level_name}) sufficiently distinct to serve as reliable evaluation points? What are the implications for comparing MLLM outputs?]*
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Placeholder summary report saved to: {report_path}")


def main():
    """Main function to orchestrate the analysis."""
    script_dir = Path(__file__).resolve().parent # experiment/human_expert/src/
    # Consistent with previous scripts, results from dependencies are in ../result/ not ../src/result
    base_input_dir = script_dir.parent / "result/analysis_results" 

    # Define input file paths
    scholar_features_path = base_input_dir / "aggregated_scholar_features.csv"
    book_features_path = base_input_dir / "aggregated_book_features.csv"
    scholar_centroids_path = base_input_dir / "aggregated_scholar_semantic_centroids.csv"
    book_centroids_path = base_input_dir / "aggregated_book_semantic_centroids.csv"

    # Ensure base output directory and subdirectories exist
    ensure_dir(BASE_OUTPUT_DIR)
    ensure_dir(SCHOLAR_FEATURE_DIST_DIR)
    ensure_dir(BOOK_FEATURE_DIST_DIR)
    
    print(f"Base output directory set to: {BASE_OUTPUT_DIR.resolve()}")


    # Load data
    try:
        df_scholar_f, df_book_f, df_scholar_c, df_book_c = load_data(
            scholar_features_path, book_features_path,
            scholar_centroids_path, book_centroids_path
        )
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found. {e}")
        print("Please ensure that `aggregate_expert_features.py` and `aggregate_expert_semantic_vectors.py` have been run successfully and their outputs are in the expected locations.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # Analyze features
    analyze_aggregated_features(df_scholar_f, "scholar", BASE_OUTPUT_DIR, SCHOLAR_FEATURE_DIST_DIR)
    analyze_aggregated_features(df_book_f, "book", BASE_OUTPUT_DIR, BOOK_FEATURE_DIST_DIR)

    # Analyze semantic centroids
    analyze_aggregated_semantic_centroids(df_scholar_c, "scholar", BASE_OUTPUT_DIR)
    analyze_aggregated_semantic_centroids(df_book_c, "book", BASE_OUTPUT_DIR)

    # Generate placeholder summary reports
    generate_summary_report_placeholder(BASE_OUTPUT_DIR, "scholar")
    generate_summary_report_placeholder(BASE_OUTPUT_DIR, "book")

    print("\\n--- Script finished ---")

if __name__ == "__main__":
    main() 