import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import ConvexHull

def visualize_vectors():
    """
    Visualizes semantic vectors of human expert and MLLM commentaries and their profile centroids
    using t-SNE for dimensionality reduction, on a single unified plot.
    Human expert profiles are shown with convex hulls. MLLM data is overlaid with its overall centroid.
    """
    script_dir = Path(__file__).resolve().parent
    base_results_path = script_dir / "../../result/analysis_results/"
    output_plot_dir = script_dir / "../result/eda_plots/"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    vectors_path = base_results_path / "all_semantic_vectors_with_profiles.csv"
    centroids_path = base_results_path / "all_profile_semantic_centroids.csv"

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
    print(f"Loaded {len(df_centroids)} profile centroids.")

    # --- 2. Prepare Vectors and Labels ---
    
    vector_dim_cols = [str(i) for i in range(1024)]

    missing_vector_cols = [col for col in vector_dim_cols if col not in df_vectors.columns]
    if missing_vector_cols:
        print(f"Error: Missing expected vector dimension columns in df_vectors: {missing_vector_cols[:5]}...")
        return
        
    try:
        individual_vectors_raw = df_vectors[vector_dim_cols].values 
        profile_column_name = 'primary_profile_or_status'
        if profile_column_name not in df_vectors.columns:
            print(f"Error: Expected profile column '{profile_column_name}' not found in vectors file: {vectors_path}")
            return
        individual_labels_raw = df_vectors[profile_column_name].tolist()
        individual_sources_raw = df_vectors['source'].tolist()
    except KeyError as e:
        print(f"Error accessing columns for individual vectors or labels: {e}. Check CSV column names.")
        return
    except Exception as e:
        print(f"Error reconstructing individual vectors: {e}")
        return

    missing_centroid_cols = [col for col in vector_dim_cols if col not in df_centroids.columns]
    if missing_centroid_cols:
        print(f"Error: Missing expected vector dimension columns in df_centroids: {missing_centroid_cols[:5]}...")
        return

    try:
        centroid_vectors_raw = df_centroids[vector_dim_cols].values
        profile_column_name_centroids = 'profile_name'
        if profile_column_name_centroids not in df_centroids.columns and df_centroids.index.name == profile_column_name_centroids:
            df_centroids = df_centroids.reset_index()
        
        if profile_column_name_centroids not in df_centroids.columns:
             print(f"Error: Expected profile column '{profile_column_name_centroids}' not found in centroids file: {centroids_path}")
             return
        centroid_labels_raw = df_centroids[profile_column_name_centroids].tolist()
        centroid_sources_raw = df_centroids['source'].tolist()
    except KeyError as e:
        print(f"Error accessing columns for centroid vectors or labels: {e}. Check CSV column names.")
        return
    except Exception as e:
        print(f"Error reconstructing centroid vectors: {e}")
        return

    # Filter out rows with NaN vectors before further processing
    nan_mask_individual = ~np.isnan(individual_vectors_raw).any(axis=1)
    individual_vectors = individual_vectors_raw[nan_mask_individual]
    individual_labels = [individual_labels_raw[i] for i, keep in enumerate(nan_mask_individual) if keep]
    individual_sources = [individual_sources_raw[i] for i, keep in enumerate(nan_mask_individual) if keep]
    
    if len(individual_vectors_raw) != len(individual_vectors):
        print(f"Warning: Dropped {len(individual_vectors_raw) - len(individual_vectors)} rows from individual_vectors due to NaNs in vector dimensions.")

    nan_mask_centroids = ~np.isnan(centroid_vectors_raw).any(axis=1)
    centroid_vectors = centroid_vectors_raw[nan_mask_centroids]
    centroid_labels = [centroid_labels_raw[i] for i, keep in enumerate(nan_mask_centroids) if keep]
    centroid_sources = [centroid_sources_raw[i] for i, keep in enumerate(nan_mask_centroids) if keep]

    if len(centroid_vectors_raw) != len(centroid_vectors):
        print(f"Warning: Dropped {len(centroid_vectors_raw) - len(centroid_vectors)} rows from centroid_vectors due to NaNs in vector dimensions. This should ideally not happen for centroids.")


    if individual_vectors.shape[0] == 0:
        print("Error: No individual vectors to process after NaN removal.")
        return
    if centroid_vectors.shape[0] == 0:
        print("Error: No centroid vectors to process after NaN removal.")
        # It's possible to proceed without centroids, but plotting will be limited.
        # For this script's purpose, centroids are expected.
        # return # Or handle gracefully if centroids are optional

    print(f"Processed individual vectors shape: {individual_vectors.shape}")
    print(f"Processed centroid vectors shape: {centroid_vectors.shape}")

    # --- 3. Merge Vectors for Joint t-SNE Transformation ---
    all_vectors = np.vstack((individual_vectors, centroid_vectors))
    
    # These labels are for the combined 'all_vectors' before t-SNE
    # Not directly used for plotting sources later, but good for consistency if needed
    # source_labels_for_all_vectors = ['individual'] * len(individual_vectors) + ['centroid'] * len(centroid_vectors)
    # profile_labels_for_all_vectors = individual_labels + centroid_labels
    # data_sources_for_all_vectors = individual_sources + centroid_sources

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
        print(f"Warning: Could not set Chinese font (SimHei). Error: {font_e}")

    fig, ax = plt.subplots(1, 1, figsize=(18, 14)) # Single plot

    # Define a comprehensive list of profiles for consistent coloring
    all_plotted_profiles = sorted(list(set(p for p in individual_labels + centroid_labels if pd.notna(p))))
    palette = sns.color_palette("husl", len(all_plotted_profiles))
    color_map = {profile: palette[i] for i, profile in enumerate(all_plotted_profiles)}
    
    # Define markers and styles
    human_point_marker = 'o'
    centroid_marker = 'X'
    human_point_alpha = 0.5
    human_point_size = 50
    centroid_point_size = 200
    centroid_edge_color = 'black'
    centroid_linewidth = 1.5
    convex_hull_alpha = 0.15

    mllm_point_marker = '^'
    mllm_point_color = 'purple' # Specific color for MLLM points
    mllm_point_alpha = 0.6
    mllm_point_size = 40 

    mllm_overall_centroid_marker = '*'
    mllm_overall_centroid_color = 'red' # Specific color for MLLM overall centroid
    mllm_overall_centroid_size = centroid_point_size + 50

    ax.set_title('Unified Semantic Space: Human Profiles (with Convex Hulls) vs. MLLM Data (t-SNE)', fontsize=18)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)

    # A. Plot Human Expert Profiles (points and convex hulls)
    # Iterate through profiles that are present in human expert data for creating hulls
    human_expert_profiles_for_hulls = sorted(list(set(p for p, s in zip(individual_labels, individual_sources) if pd.notna(p) and s == 'human_expert')))

    for profile_name_iter in human_expert_profiles_for_hulls:
        current_profile_points_x = []
        current_profile_points_y = []
        for i, (label_ind, source_ind) in enumerate(zip(individual_labels, individual_sources)):
            if label_ind == profile_name_iter and source_ind == 'human_expert':
                current_profile_points_x.append(individual_vectors_2d[i, 0])
                current_profile_points_y.append(individual_vectors_2d[i, 1])
        
        if current_profile_points_x:
            ax.scatter(current_profile_points_x, current_profile_points_y,
                       label=f"{profile_name_iter} (Human)", # Clarify it's human data
                       alpha=human_point_alpha, s=human_point_size,
                       color=color_map.get(profile_name_iter, 'grey'), # Fallback color
                       marker=human_point_marker)

            if len(current_profile_points_x) >= 3:
                profile_points_2d_np = np.array(list(zip(current_profile_points_x, current_profile_points_y)))
                try:
                    hull = ConvexHull(profile_points_2d_np)
                    ax.fill(profile_points_2d_np[hull.vertices,0], profile_points_2d_np[hull.vertices,1],
                            color=color_map.get(profile_name_iter, 'grey'), alpha=convex_hull_alpha,
                            label='_nolegend_') # Convex hulls don't need separate legend items
                except Exception as e: # More specific: from scipy.spatial import QhullError
                    print(f"Could not compute or draw convex hull for human profile {profile_name_iter}: {e}")

    # B. Plot Human Expert Centroids (and 'both' if applicable)
    for i, (prof_label, src) in enumerate(zip(centroid_labels, centroid_sources)):
        if pd.isna(prof_label): continue
        if src == 'human_expert' or src == 'both': 
            ax.scatter(centroid_vectors_2d[i, 0], centroid_vectors_2d[i, 1],
                       marker=centroid_marker, s=centroid_point_size, color=color_map.get(prof_label, 'grey'),
                       edgecolor=centroid_edge_color, linewidth=centroid_linewidth,
                       label=f"{prof_label} Centroid (Human Source: {src})", alpha=1.0)
            ax.text(centroid_vectors_2d[i, 0] + 0.05, centroid_vectors_2d[i, 1] + 0.05, 
                    prof_label, fontsize=12, color=color_map.get(prof_label, 'grey'), weight='bold')

    # C. Plot MLLM Data Points
    mllm_individual_points_x = []
    mllm_individual_points_y = []
    for i, (vec_2d, source_ind) in enumerate(zip(individual_vectors_2d, individual_sources)):
        if source_ind == 'mllm':
            mllm_individual_points_x.append(vec_2d[0])
            mllm_individual_points_y.append(vec_2d[1])
            
    if mllm_individual_points_x:
        ax.scatter(mllm_individual_points_x, mllm_individual_points_y,
                   label="MLLM Data Points",
                   alpha=mllm_point_alpha, s=mllm_point_size,
                   color=mllm_point_color,
                   marker=mllm_point_marker)

        # D. Calculate and Plot MLLM Overall Centroid
        mllm_points_for_centroid_np = np.array(list(zip(mllm_individual_points_x, mllm_individual_points_y)))
        if mllm_points_for_centroid_np.shape[0] > 0:
            mllm_overall_centroid_2d = np.mean(mllm_points_for_centroid_np, axis=0)
            ax.scatter(mllm_overall_centroid_2d[0], mllm_overall_centroid_2d[1],
                       marker=mllm_overall_centroid_marker, s=mllm_overall_centroid_size,
                       color=mllm_overall_centroid_color,
                       edgecolor=centroid_edge_color, linewidth=centroid_linewidth,
                       label="MLLM Overall Centroid", alpha=1.0)
            ax.text(mllm_overall_centroid_2d[0] + 0.05, mllm_overall_centroid_2d[1] + 0.05,
                    "MLLM Overall", fontsize=12, color=mllm_overall_centroid_color, weight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels/handles if any, preserving order
    by_label = {}
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle
            new_handles.append(handle)
            new_labels.append(label)
            
    fig.legend(new_handles, new_labels, title="Legend", 
               fontsize=10, title_fontsize=12, loc='center left', bbox_to_anchor=(1.01, 0.5)) 
    
    # Adjust layout to make space for legend
    fig.tight_layout(rect=[0, 0, 0.86, 0.95]) # rect right boundary might need adjustment based on legend width

    # --- 7. Save Plot ---
    output_file_path = output_plot_dir / "unified_semantic_space_plan_b_tsne.png" # New filename for Plan B
    try:
        fig.savefig(output_file_path, bbox_inches='tight') # Added bbox_inches for potentially wide legend
        print(f"Plot saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # plt.show() 

if __name__ == "__main__":
    visualize_vectors()