import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from math import pi # For radar chart

# Define a color palette for profiles for consistency
# Using a dictionary for easy lookup by profile name (without "score_" or "proportion_")
PROFILE_COLORS = {
    "Comprehensive_Analyst": "#1f77b4",  # Muted Blue
    "Historically_Focused": "#ff7f0e",  # Safety Orange
    "Technique_and_Style_Focused": "#2ca02c",  # Cooked Asparagus Green
    "Theory_and_Comparison_Focused": "#d62728",  # Brick Red
    "General_Descriptive_Profile": "#9467bd",  # Muted Purple
    "Lacks_Specific_Focus": "#8c564b",  # Chestnut Brown
    "Other": "#e377c2",  # Raspberry Sorbet Pink (fallback)
    # Add more if other specific primary profiles emerge
}

# Helper to get color, defaulting if profile unknown
def get_profile_color(profile_name):
    return PROFILE_COLORS.get(profile_name, PROFILE_COLORS["Other"])

def plot_stacked_bar_proportions(ax, data_human, data_mllm, profiles, file_id_to_plot):
    """
    Plots stacked bar charts for human vs. MLLM profile proportions for a given file_id.
    Assumes data_human and data_mllm are Series containing proportion_ProfileName columns.
    'profiles' is a list of profile names (e.g., "Comprehensive_Analyst").
    """
    n_profiles = len(profiles)
    bar_width = 0.35
    index = np.arange(n_profiles)

    human_proportions = [data_human.get(f'proportion_{p}', 0) for p in profiles]
    mllm_proportions = [data_mllm.get(f'proportion_{p}', 0) for p in profiles]

    # For stacking, we need cumulative sums
    human_bottom = np.zeros(len(profiles))
    mllm_bottom = np.zeros(len(profiles))

    # Simplified: just two bars, one for human, one for MLLM, each bar represents the text.
    # Each segment in the bar is a profile's proportion.

    # Human Bar
    left_pos_human = 0 # Position of the human bar
    current_bottom_human = 0
    for i, profile_name in enumerate(profiles):
        proportion = data_human.get(f'proportion_{profile_name}', 0)
        ax.bar(left_pos_human, proportion, bar_width, bottom=current_bottom_human,
               color=get_profile_color(profile_name), label=profile_name if i == 0 else "") # Label only once per profile
        current_bottom_human += proportion
    
    # MLLM Bar
    left_pos_mllm = bar_width + 0.1 # Position of the MLLM bar, with a small gap
    current_bottom_mllm = 0
    for i, profile_name in enumerate(profiles):
        proportion = data_mllm.get(f'proportion_{profile_name}', 0)
        ax.bar(left_pos_mllm, proportion, bar_width, bottom=current_bottom_mllm,
               color=get_profile_color(profile_name)) # No duplicate labels needed from MLLM bar
        current_bottom_mllm += proportion

    ax.set_title(f'Proportions for: {file_id_to_plot}')
    ax.set_xticks([left_pos_human, left_pos_mllm])
    ax.set_xticklabels(['Human', 'MLLM'])
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1) # Proportions sum to 1

    # Create a single legend for profiles if not too cluttered
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles)) # Remove duplicate labels for legend
    # ax.legend(by_label.values(), by_label.keys(), title='Profiles', bbox_to_anchor=(1.05, 1), loc='upper left')


def plot_radar_proportions(ax, data_human, data_mllm, profiles, file_id_to_plot):
    """
    Plots radar charts for human vs. MLLM profile proportions.
    'profiles' is a list of profile names (e.g., "Comprehensive_Analyst").
    """
    num_vars = len(profiles)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the plot

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(profiles)
    ax.set_rlabel_position(0) # Move radial labels away from plotted line
    ax.set_yticks([0.2, 0.4, 0.6, 0.8]) # Adjust as needed
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_ylim(0, 1)

    # Human data
    human_values = [data_human.get(f'proportion_{p}', 0) for p in profiles]
    human_values += human_values[:1] # Close the plot
    ax.plot(angles, human_values, linewidth=1, linestyle='solid', label='Human', color=PROFILE_COLORS["Comprehensive_Analyst"]) # Example color
    ax.fill(angles, human_values, PROFILE_COLORS["Comprehensive_Analyst"], alpha=0.1)

    # MLLM data
    mllm_values = [data_mllm.get(f'proportion_{p}', 0) for p in profiles]
    mllm_values += mllm_values[:1] # Close the plot
    ax.plot(angles, mllm_values, linewidth=1, linestyle='solid', label='MLLM', color=PROFILE_COLORS["Historically_Focused"]) # Example color
    ax.fill(angles, mllm_values, PROFILE_COLORS["Historically_Focused"], alpha=0.1)

    ax.set_title(f'Radar for: {file_id_to_plot}', size='small', y=1.1)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


def plot_dot_proportions(ax, data_human, data_mllm, profiles, file_id_to_plot):
    """
    Plots dot plots for human vs. MLLM profile proportions.
    'profiles' is a list of profile names.
    """
    y_pos = np.arange(len(profiles))

    human_proportions = [data_human.get(f'proportion_{p}', 0) for p in profiles]
    mllm_proportions = [data_mllm.get(f'proportion_{p}', 0) for p in profiles]

    ax.scatter(human_proportions, y_pos, color=PROFILE_COLORS["Comprehensive_Analyst"], label='Human', s=50, alpha=0.7, marker='o')
    ax.scatter(mllm_proportions, y_pos + 0.1, color=PROFILE_COLORS["Historically_Focused"], label='MLLM', s=50, alpha=0.7, marker='x') # Offset MLLM slightly

    ax.set_yticks(y_pos)
    ax.set_yticklabels(profiles)
    ax.set_xlabel('Proportion')
    ax.set_title(f'Dots for: {file_id_to_plot}')
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.legend()


def plot_convex_hulls_or_kde(ax, df, profiles_list, x_col='tsne_x', y_col='tsne_y', plot_type='hull'):
    """
    Plots convex hulls or KDE for different source_types and primary_profiles.
    'profiles_list' includes "Lacks_Specific_Focus".
    """
    ax.set_title(f'{plot_type.capitalize()} Plot of Profile Clusters ({x_col.split("_")[0].upper()})')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    legend_handles = []

    for source_type in df['source_type'].unique():
        for profile_name in profiles_list: # Iterate through defined profiles including Lacks_Specific_Focus
            # Filter data for current source_type and profile
            subset = df[(df['source_type'] == source_type) & (df['primary_profile_by_proportion'] == profile_name)]
            
            if subset.empty:
                continue

            color = get_profile_color(profile_name)
            
            # Scatter plot for all points in the subset
            scatter = ax.scatter(subset[x_col], subset[y_col], color=color, 
                                 marker='o' if source_type == 'human' else 'x', # Different markers
                                 s=20, alpha=0.5, 
                                 label=f'{source_type.capitalize()} - {profile_name}' if not any(lh.get_label() == f'{source_type.capitalize()} - {profile_name}' for lh in legend_handles) else "")

            if plot_type == 'hull' and len(subset) >= 3:
                try:
                    points = subset[[x_col, y_col]].values
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1], color=color, 
                                linestyle='-' if source_type == 'human' else '--', # Different linestyles
                                linewidth=1.5, alpha=0.8)
                except Exception as e:
                    print(f"Could not compute hull for {source_type} - {profile_name}: {e}")
            elif plot_type == 'kde':
                try:
                    sns.kdeplot(data=subset, x=x_col, y=y_col, color=color, ax=ax,
                                levels=3, thresh=0.1, alpha=0.2,
                                fill=True if source_type == 'human' else False, # Fill for human, lines for MLLM
                                linestyle='-' if source_type == 'human' else '--')
                except Exception as e:
                    print(f"Could not compute KDE for {source_type} - {profile_name}: {e}")
            
            # Collect unique legend handles
            if scatter.get_label(): # Only add if label is not empty (to avoid duplicates)
                 legend_handles.append(scatter)

    # Create a consolidated legend
    # Sort handles to group human/mllm or by profile if desired
    # legend_handles.sort(key=lambda h: h.get_label())
    # ax.legend(handles=legend_handles, title="Source - Profile", bbox_to_anchor=(1.05, 1), loc='upper left')


def main():
    script_dir = Path(__file__).resolve().parent
    base_result_dir = script_dir.parent / "result" # experiment/human_expert/result
    input_csv_path = base_result_dir / "analysis_results" / "dimensionality_reduction_with_profile_scores_human_and_gemini.csv"
    output_plot_dir = base_result_dir / "eda_plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv_path.exists():
        print(f"Error: Input CSV not found at {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    # Define the five core profiles for proportion display (names without 'proportion_' prefix)
    core_profiles_for_display = [
        "Comprehensive_Analyst",
        "Historically_Focused",
        "Technique_and_Style_Focused",
        "Theory_and_Comparison_Focused",
        "General_Descriptive_Profile"
    ]
    
    # Profiles for main plot legend (includes Lacks_Specific_Focus)
    all_profiles_for_main_plot_legend = core_profiles_for_display + ["Lacks_Specific_Focus"]


    # --- Select Sample Texts for Subplots ---
    # Try to find a file_id that exists for both human and mllm
    sample_file_id = None
    if 'file_id' in df.columns:
        # Count occurrences of each file_id by source_type
        file_id_counts = df.groupby('file_id')['source_type'].nunique()
        both_sources_ids = file_id_counts[file_id_counts > 1].index.tolist()
        if both_sources_ids:
            sample_file_id = both_sources_ids[0] # Pick the first one
        elif not df.empty: # Pick any if no overlap
            sample_file_id = df['file_id'].iloc[0] 

    data_human_sample = pd.Series(dtype=float)
    data_mllm_sample = pd.Series(dtype=float)

    if sample_file_id:
        human_row = df[(df['file_id'] == sample_file_id) & (df['source_type'] == 'human')]
        mllm_row = df[(df['file_id'] == sample_file_id) & (df['source_type'] == 'mllm')]
        if not human_row.empty:
            data_human_sample = human_row.iloc[0]
        if not mllm_row.empty:
            data_mllm_sample = mllm_row.iloc[0]
        print(f"Using sample_file_id: {sample_file_id} for subplots.")
    else:
        print("Could not find a suitable sample file_id for subplots or DataFrame is empty.")


    # --- Create Composite Visualization ---
    # Option 1: Main plot (t-SNE Hulls) + 3 small subplots (Bar, Radar, Dot) for one sample
    fig1, axs1 = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw={'height_ratios': [3, 1.5]})
    fig1.suptitle('Composite Profile Analysis (t-SNE Hulls & Sample Proportions)', fontsize=16)

    # Main plot: t-SNE Convex Hulls
    plot_convex_hulls_or_kde(axs1[0, 0], df, all_profiles_for_main_plot_legend, x_col='tsne_x', y_col='tsne_y', plot_type='hull')
    
    # Placeholder for second main plot (e.g., UMAP Hulls or t-SNE KDE)
    # For now, let's use the second large slot for a different type or UMAP version of the main plot
    plot_convex_hulls_or_kde(axs1[0, 1], df, all_profiles_for_main_plot_legend, x_col='umap_x', y_col='umap_y', plot_type='hull')
    axs1[0,1].set_title('Convex Hull Plot of Profile Clusters (UMAP)')


    if sample_file_id and not data_human_sample.empty and not data_mllm_sample.empty:
        plot_stacked_bar_proportions(axs1[1, 0], data_human_sample, data_mllm_sample, core_profiles_for_display, sample_file_id)
        
        # Radar needs a polar subplot
        # Create a new axis for radar if needed or use an existing one if it's made polar
        # For simplicity, we will replace one of the small subplot axes with a polar one.
        # This might require adjusting the plt.subplots() call if we want to keep all three small ones.
        # Let's try adding it to axs1[1,1] for now.
        # fig1.delaxes(axs1[1,1]) # Remove the Cartesian axis
        # ax_radar = fig1.add_subplot(2, 2, 4, projection='polar') # Add polar axis in its place
        # plot_radar_proportions(ax_radar, data_human_sample, data_mllm_sample, core_profiles_for_display, sample_file_id)
        # axs1[1,1].set_title("Radar Chart (Sample)") # Add title to the original axis position conceptually

        # Dot plot
        # plot_dot_proportions(axs1[1, 2], data_human_sample, data_mllm_sample, core_profiles_for_display, sample_file_id) # if 2x3 layout
        # For 2x2 layout, we might only show two small plots or need to rethink layout.
        # Let's put dot plot in the second small slot for now.
        plot_dot_proportions(axs1[1, 1], data_human_sample, data_mllm_sample, core_profiles_for_display, sample_file_id)
        axs1[1,1].set_title("Dot Plot (Sample)")


    else:
        axs1[1,0].text(0.5, 0.5, 'No/Insufficient sample data for proportion subplots', ha='center', va='center')
        axs1[1,1].text(0.5, 0.5, 'No/Insufficient sample data for proportion subplots', ha='center', va='center')

    # Add a single consolidated legend for the main plots if possible, or for subplots
    # For main plots, a single legend is complex due to combined source/profile.
    # Consider separate legends or placing them carefully.
    handles_main, labels_main = axs1[0,0].get_legend_handles_labels() # Get from one of the main plots
    # Filter unique labels for main plot legend
    by_label_main = dict(zip(labels_main, handles_main))
    fig1.legend(by_label_main.values(), by_label_main.keys(), title="Source - Profile (Hulls)", loc='center right', bbox_to_anchor=(0.98, 0.7))


    # Legend for bar plot (profiles)
    # We need to collect labels from one of the bar plots (e.g., human bar)
    # Create dummy plots for legend if direct collection is hard
    legend_elements_bar = [plt.Rectangle((0,0),1,1, color=get_profile_color(p_name)) for p_name in core_profiles_for_display]
    axs1[1,0].legend(legend_elements_bar, core_profiles_for_display, title="Profiles (Bar)", loc="upper right", fontsize='small')


    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust rect to make space for suptitle and legend
    output_path1 = output_plot_dir / "composite_analysis_hulls.png"
    plt.savefig(output_path1)
    print(f"Composite plot with Hulls saved to {output_path1}")
    plt.close(fig1)


    # --- Option 2: Main plot (t-SNE KDE) + Radar for sample ---
    fig2, axs2 = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3,1]}) # One main, one small for radar
    fig2.suptitle('Composite Profile Analysis (t-SNE KDE & Sample Radar)', fontsize=16)

    # Main plot: t-SNE KDE
    plot_convex_hulls_or_kde(axs2[0], df, all_profiles_for_main_plot_legend, x_col='tsne_x', y_col='tsne_y', plot_type='kde')
    
    # Subplot: Radar
    if sample_file_id and not data_human_sample.empty and not data_mllm_sample.empty:
        # Need to make this subplot polar
        fig2.delaxes(axs2[1]) # Remove the Cartesian axis
        ax_radar_fig2 = fig2.add_subplot(1, 2, 2, projection='polar') # Add polar axis in its place
        plot_radar_proportions(ax_radar_fig2, data_human_sample, data_mllm_sample, core_profiles_for_display, sample_file_id)
        ax_radar_fig2.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), fontsize='small') # Adjust legend for radar
    else:
        axs2[1].text(0.5, 0.5, 'No/Insufficient sample data for radar subplot', ha='center', va='center')

    handles_kde, labels_kde = axs2[0].get_legend_handles_labels()
    by_label_kde = dict(zip(labels_kde, handles_kde))
    fig2.legend(by_label_kde.values(), by_label_kde.keys(), title="Source - Profile (KDE)", loc='center right', bbox_to_anchor=(0.98, 0.6))


    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    output_path2 = output_plot_dir / "composite_analysis_kde_radar.png"
    plt.savefig(output_path2)
    print(f"Composite plot with KDE and Radar saved to {output_path2}")
    plt.close(fig2)

    print("Visualization script finished.")

if __name__ == '__main__':
    main() 