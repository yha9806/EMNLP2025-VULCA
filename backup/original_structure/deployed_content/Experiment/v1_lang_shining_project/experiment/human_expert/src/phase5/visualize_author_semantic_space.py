import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_author_vectors():
    """
    Visualizes semantic vectors of human expert and MLLM commentaries (labeled by author)
    and their author centroids using t-SNE for dimensionality reduction.
    """
    script_dir = Path(__file__).resolve().parent
    base_results_path = script_dir / "../../result/analysis_results/"
    output_plot_dir = script_dir / "../result/eda_plots/"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    vectors_path = base_results_path / "all_author_semantic_vectors.csv"
    centroids_path = base_results_path / "all_author_semantic_centroids.csv"

    print(f"Loading individual vectors from: {vectors_path}")
    try:
        df_vectors = pd.read_csv(vectors_path)
        print(f"Columns in df_vectors: {df_vectors.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Individual vectors file not found at {vectors_path}")
        return
    print(f"Loaded {len(df_vectors)} individual vectors.")

    print(f"Loading centroids from: {centroids_path}")
    try:
        df_centroids = pd.read_csv(centroids_path)
        print(f"Columns in df_centroids: {df_centroids.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Centroids file not found at {centroids_path}")
        return
    print(f"Loaded {len(df_centroids)} author centroids.")

    # --- 2. Prepare Vectors and Labels ---
    
    # Determine vector dimension columns (assuming they are '0' to '1023')
    # For BGE-large-zh-v1.5, the dimension is 1024. Columns '0' through '1023'.
    vector_dim_cols = [str(i) for i in range(1024)]

    # Check if all expected dimension columns exist in df_vectors
    missing_vector_cols = [col for col in vector_dim_cols if col not in df_vectors.columns]
    if missing_vector_cols:
        print(f"Error: Missing expected vector dimension columns in df_vectors: {missing_vector_cols[:5]}...") # Print first 5 missing
        print(f"Available columns in vectors file: {df_vectors.columns.tolist()}")
        return
        
    # Reconstruct individual vectors
    try:
        individual_vectors = df_vectors[vector_dim_cols].values 
        individual_labels = df_vectors['author_label'].tolist()
        individual_sources = df_vectors['source'].tolist()  # Extract source information
    except KeyError as e:
        print(f"Error accessing columns for individual vectors or labels: {e}. Check CSV column names. Expected 'author_label' and 'source'.")
        return
    except Exception as e:
        print(f"Error reconstructing individual vectors: {e}")
        return

    # Check if all expected dimension columns exist in df_centroids
    missing_centroid_cols = [col for col in vector_dim_cols if col not in df_centroids.columns]
    if missing_centroid_cols:
        print(f"Error: Missing expected vector dimension columns in df_centroids: {missing_centroid_cols[:5]}...")
        return

    # Reconstruct centroid vectors
    try:
        centroid_vectors = df_centroids[vector_dim_cols].values
        # Assuming 'author_label' is the index in the centroids CSV
        author_label_col_centroids = 'author_label' # This should be the column name for author labels in centroids CSV
        if author_label_col_centroids not in df_centroids.columns and df_centroids.index.name == author_label_col_centroids:
            # If author_label is the index, reset it to make it a column
            df_centroids = df_centroids.reset_index()
        
        if author_label_col_centroids not in df_centroids.columns:
             print(f"Error: Expected author label column '{author_label_col_centroids}' not found in centroids file: {centroids_path}")
             print(f"Available columns in centroids file: {df_centroids.columns.tolist()}")
             return
        centroid_labels = df_centroids[author_label_col_centroids].tolist()
        centroid_sources = df_centroids['source'].tolist()  # Extract source information for centroids
    except KeyError as e:
        print(f"Error accessing columns for centroid vectors or labels: {e}. Check CSV column names. Expected '{author_label_col_centroids}' and 'source'.")
        return
    except Exception as e:
        print(f"Error reconstructing centroid vectors: {e}")
        return

    # Ensure no NaN values in the reconstructed vectors, which could cause issues for t-SNE
    if np.isnan(individual_vectors).any():
        print("Warning: NaN values found in individual_vectors. Attempting to drop rows with NaNs.")
        mask = df_vectors[vector_dim_cols].notna().all(axis=1)
        df_vectors_cleaned = df_vectors[mask]
        individual_vectors = df_vectors_cleaned[vector_dim_cols].values
        individual_labels = df_vectors_cleaned['author_label'].tolist()
        individual_sources = df_vectors_cleaned['source'].tolist()
        print(f"Retained {len(individual_vectors)} individual vectors after NaN removal.")
        if len(individual_vectors) == 0:
            print("Error: No individual vectors left after NaN removal.")
            return
            
    if np.isnan(centroid_vectors).any():
        print("Error: NaN values found in centroid_vectors. Please check centroid generation script.")
        return

    if individual_vectors.shape[0] == 0:
        print("Error: No individual vectors to process after preparation.")
        return
    if centroid_vectors.shape[0] == 0:
        print("Error: No centroid vectors to process after preparation.")
        return

    print(f"Processed individual vectors shape: {individual_vectors.shape}")
    print(f"Processed centroid vectors shape: {centroid_vectors.shape}")

    # --- 3. Merge Vectors for Joint t-SNE Transformation ---
    all_vectors = np.vstack((individual_vectors, centroid_vectors))
    
    source_labels = ['individual'] * len(individual_vectors) + ['centroid'] * len(centroid_vectors)
    # Author labels for all vectors including centroids
    author_labels_all = individual_labels + centroid_labels
    # Source information for all vectors including centroids
    data_sources_all = individual_sources + centroid_sources

    # --- 4. Execute Dimensionality Reduction (t-SNE) ---
    print("Performing t-SNE reduction...")
    n_samples = len(all_vectors)
    perplexity_value = min(30, n_samples - 1) if n_samples > 1 else 0
    
    if perplexity_value <= 0:
        print(f"Error: Not enough samples ({n_samples}) for t-SNE. Need at least 2 samples.")
        return
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=1000, metric='cosine')
    try:
        all_vectors_2d = tsne.fit_transform(all_vectors)
    except ValueError as e:
        print(f"Error during t-SNE: {e}")
        print(f"This might happen if the number of samples is too small for the chosen perplexity.")
        print(f"Number of samples: {len(all_vectors)}")
        return

    # --- 5. Separate Reduced Vectors ---
    individual_vectors_2d = all_vectors_2d[:len(individual_vectors)]
    centroid_vectors_2d = all_vectors_2d[len(individual_vectors):]

    # --- 6. Plotting ---
    print("Plotting vectors...")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as font_e:
        print(f"Warning: Could not set SimHei font. Error: {font_e}")

    plt.figure(figsize=(16, 12))
    
    # Use unique author labels for coloring
    unique_labels = sorted(list(set(author_labels_all)))
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    # Define markers for different data sources
    marker_map = {
        'human_expert': 'o',  # circle for human expert
        'mllm': 's',          # square for mllms
        'both': 'd'           # diamond for both
    }
    
    # Plot individual vectors with different markers based on source
    for current_label_val in unique_labels:
        # Plot human expert vectors
        human_indices = [i for i, (label, source) in enumerate(zip(individual_labels, individual_sources)) 
                         if label == current_label_val and source == 'human_expert']
        if human_indices:
            plt.scatter(individual_vectors_2d[human_indices, 0], individual_vectors_2d[human_indices, 1],
                        label=f"{current_label_val} (Human Expert)", alpha=0.7, s=50, 
                        color=color_map.get(current_label_val), marker=marker_map['human_expert'])
        
        # Plot mllm vectors
        mllm_indices = [i for i, (label, source) in enumerate(zip(individual_labels, individual_sources)) 
                         if label == current_label_val and source == 'mllm']
        if mllm_indices:
            plt.scatter(individual_vectors_2d[mllm_indices, 0], individual_vectors_2d[mllm_indices, 1],
                        label=f"{current_label_val} (MLLM)", alpha=0.7, s=50, 
                        color=color_map.get(current_label_val), marker=marker_map['mllm'])

    # Plot centroids with different markers based on source
    for i, (label_val, source) in enumerate(zip(centroid_labels, centroid_sources)):
        plt.scatter(centroid_vectors_2d[i, 0], centroid_vectors_2d[i, 1],
                    marker='X', s=200, color=color_map.get(label_val, 'black'),
                    edgecolor='black', linewidth=1.5,
                    label=f"{label_val} Centroid ({source.capitalize()})")
        # Add text annotation for centroids
        plt.text(centroid_vectors_2d[i, 0] + 0.05, centroid_vectors_2d[i, 1] + 0.05, 
                 label_val, fontsize=12, color=color_map.get(label_val, 'black'), weight='bold')

    plt.title('Author Semantic Space Visualization (t-SNE Dimensionality Reduction)', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels while preserving the order
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Author & Source", fontsize=10, title_fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # --- 7. Save Plot ---
    output_file_path = output_plot_dir / "combined_author_semantic_space_visualization_tsne.png"
    try:
        plt.savefig(output_file_path)
        print(f"Plot saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show() # Commented out for automated execution

if __name__ == "__main__":
    visualize_author_vectors() 