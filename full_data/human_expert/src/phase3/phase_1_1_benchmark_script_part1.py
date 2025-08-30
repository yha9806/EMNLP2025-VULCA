import pandas as pd
import ast
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from collections import defaultdict
from scipy import stats
import sys
import traceback

# UMAP is optional, attempt import and handle if not found
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn package not found. UMAP visualization will be skipped.")

def main():
    # Define paths relative to the script location
    script_dir = Path(__file__).parent
    base_result_dir = script_dir / "../../result" # This correctly points to v1_lang_shining_project/experiment/human_expert/result
    
    # Path to the human expert CSV file
    input_csv_path_human = base_result_dir / "human_expert_features_consolidated.csv"
    
    # Path to the Gemini 2.5 Pro CSV file - relative to workspace root
    # Workspace root is script_dir.parents[3] if script is in v1_lang_shining_project/experiment/human_expert/src/phase3
    # Correcting workspace_root to point to the actual workspace root "v1_lang_shining_project"
    # Script path: v1_lang_shining_project/experiment/human_expert/src/phase3
    # parent[0] -> src
    # parent[1] -> human_expert
    # parent[2] -> experiment
    # parent[3] -> v1_lang_shining_project (This should be the root for constructing path to MLLMS data)
    workspace_project_root = script_dir.parents[3] 
    input_csv_path_gemini = workspace_project_root / "experiment/MLLMS/result/mllms_features_consolidated.csv"
    
    output_analysis_dir = base_result_dir / "analysis_results"
    output_eda_plots_dir = base_result_dir / "eda_plots"
    
    # Create output directories if they don't exist
    output_analysis_dir.mkdir(parents=True, exist_ok=True)
    output_eda_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Updated output file names
    eda_summary_csv_path = output_analysis_dir / "eda_summary_statistics_human_gemini.csv"
    dim_reduction_coords_csv_path = output_analysis_dir / "features_dimensionality_reduction_coords_human_gemini.csv"
    tsne_plot_path = output_eda_plots_dir / "features_tsne_visualization_human_gemini.png"
    umap_plot_path = output_eda_plots_dir / "features_umap_visualization_human_gemini.png"

    # Checklist 2.1: Define debug log path
    debug_log_path = workspace_project_root.parent / "script_debug_output.txt"

    # Checklist 3.1: Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_handle = None # Initialize to ensure it exists for finally block

    # Checklist 3.2, 3.3, 3.4, 3.7: Implement try...finally for redirection
    try:
        log_file_handle = open(debug_log_path, 'w', encoding='utf-8')
        sys.stdout = log_file_handle
        sys.stderr = log_file_handle

        # --- 1. 数据加载与预处理 ---
        print(f"Loading HUMAN data from {input_csv_path_human}...")
        if not input_csv_path_human.exists():
            print(f"Warning: Input HUMAN CSV not found at {input_csv_path_human}. Creating empty DataFrame.")
            df_human = pd.DataFrame()
        else:
            df_human = pd.read_csv(input_csv_path_human)
            df_human['source_type'] = 'human'
            df_human['persona_id'] = 'human_expert' # Assign persona_id for human data
            if 'stance_score' in df_human.columns:
                df_human.rename(columns={'stance_score': 'score_stance'}, inplace=True)
            elif 'score_stance' not in df_human.columns: # If neither exists, add placeholder for concat
                df_human['score_stance'] = pd.NA


        print(f"Loading GEMINI data from {input_csv_path_gemini}...")
        if not input_csv_path_gemini.exists():
            print(f"Warning: Input GEMINI CSV not found at {input_csv_path_gemini}. Creating empty DataFrame.")
            df_gemini = pd.DataFrame()
        else:
            df_gemini = pd.read_csv(input_csv_path_gemini)
            df_gemini['source_type'] = 'mllm'
            # --- Add persona_id for gemini data ---
            def get_persona_id_from_file_id(file_id):
                if pd.isna(file_id) or not isinstance(file_id, str):
                    return 'gemini_unknown_persona'
                try:
                    # Normalize path for consistent separator, then split
                    # Path(file_id).parts will also work if file_id is a valid path string
                    # Using os.path.normpath and os.sep for robustness
                    normalized_path = os.path.normpath(file_id)
                    parts = normalized_path.split(os.sep)
                    if len(parts) > 1: # If there is at least one directory level
                        return parts[0]
                    else: # If it's just a filename or no separator found after normalization
                        # This case might indicate a baseline file not in a persona subfolder
                        # or an unexpected file_id format.
                        # We can derive a more specific baseline name if needed, e.g., from filename itself.
                        return 'gemini_baseline' # Or parts[0] which would be the filename
                except Exception:
                    return 'gemini_parse_error'

            if 'file_id' in df_gemini.columns:
                df_gemini['persona_id'] = df_gemini['file_id'].apply(get_persona_id_from_file_id)
            else:
                print("Warning: 'file_id' column not found in Gemini data. Cannot derive 'persona_id'. Setting to 'gemini_unknown_persona'.")
                df_gemini['persona_id'] = 'gemini_unknown_persona'
            # --- End persona_id for gemini data ---
            if 'stance_score' in df_gemini.columns:
                df_gemini.rename(columns={'stance_score': 'score_stance'}, inplace=True)
            elif 'score_stance' not in df_gemini.columns: # If neither exists, add placeholder for concat
                df_gemini['score_stance'] = pd.NA

        # Combine DataFrames
        if df_human.empty and df_gemini.empty:
            print("Error: Both HUMAN and GEMINI data files are missing or empty. Exiting.")
            return
        elif df_human.empty:
            print("Warning: HUMAN data is empty. Proceeding with GEMINI data only.")
            df = df_gemini
            if 'persona_id' not in df.columns: # Ensure persona_id is present even if only human data
                df['persona_id'] = 'human_expert'
        elif df_gemini.empty:
            print("Warning: GEMINI data is empty. Proceeding with HUMAN data only.")
            df = df_human
            if 'persona_id' not in df.columns: # Ensure persona_id is present even if only human data
                df['persona_id'] = 'human_expert'
        else:
            # Ensure 'features' and 'quality' columns exist in both before concat for smooth merge
            for temp_df, name in [(df_human, "human"), (df_gemini, "gemini")]:
                if 'features' not in temp_df.columns:
                    print(f"Adding missing 'features' column to {name} data.")
                    temp_df['features'] = pd.Series([{} for _ in range(len(temp_df))])
                if 'quality' not in temp_df.columns:
                    print(f"Adding missing 'quality' column to {name} data.")
                    temp_df['quality'] = pd.Series([{} for _ in range(len(temp_df))])
            
            print("Combining human and gemini data...")
            df = pd.concat([df_human, df_gemini], ignore_index=True, sort=False)
        
        # Convert string representations of dictionaries to actual dictionaries
        for col in ['features', 'quality']:
            if col in df.columns:
                try:
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and isinstance(x, str) else x)
                except Exception as e:
                    print(f"Warning: Could not parse column {col}. Error: {e}. Skipping conversion for this column.")
                    df[col] = df[col].apply(lambda x: {} if pd.notnull(x) else {}) # Fallback to empty dict
            else:
                print(f"Warning: Expected column '{col}' not found in the CSV. It will be treated as empty.")
                df[col] = pd.Series([{} for _ in range(len(df))])


        print("Data loaded and preprocessed.")

        # --- 2. 探索性数据分析 (EDA) ---
        print("Performing EDA...")
        eda_results = {}

        # Stance distribution
        if 'stance_label_en' in df.columns:
            eda_results['stance_distribution'] = df['stance_label_en'].value_counts(normalize=True).to_dict()
            # EDA: Distribution by source type
            eda_results['source_type_distribution'] = df['source_type'].value_counts(normalize=True).to_dict()
            # --- EDA: Distribution by persona_id (NEW) ---
            if 'persona_id' in df.columns:
                eda_results['persona_id_distribution'] = df['persona_id'].value_counts(normalize=True).to_dict()
                # Further stance distribution by persona_id
                eda_results['stance_distribution_by_persona'] = df.groupby('persona_id')['stance_label_en'].value_counts(normalize=True).unstack(fill_value=0).to_dict('index')
            # --- End EDA by persona_id ---
        else:
            print("Warning: 'stance_label_en' column not found. Skipping stance distribution analysis.")
            eda_results['stance_distribution'] = {}

        # --- Prepare DataFrame for detailed statistical analysis of features and qualities ---
        # Flatten features and quality scores into separate columns
        # First, get all unique feature and quality keys
        all_feature_keys = set()
        if 'features' in df.columns:
            for item_dict_list in df['features'].dropna(): #dropna to avoid issues with potential NaN rows
                if isinstance(item_dict_list, dict):
                     all_feature_keys.update(item_dict_list.keys())
        all_quality_keys = set()
        if 'quality' in df.columns:
            for item_dict_list in df['quality'].dropna():
                if isinstance(item_dict_list, dict):
                    all_quality_keys.update(item_dict_list.keys())

        # Create new columns for each feature and quality score
        # Prefixing with feature_ or quality_ to avoid potential name clashes
        feature_score_cols = []
        for key in sorted(list(all_feature_keys)):
            col_name = f"feature_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}"
            feature_score_cols.append(col_name)
            df[col_name] = df['features'].apply(lambda x: x.get(key, {}).get('score') if isinstance(x, dict) and isinstance(x.get(key), dict) else (x.get(key) if isinstance(x, dict) and isinstance(x.get(key), (int, float)) else pd.NA))
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        quality_score_cols = []
        for key in sorted(list(all_quality_keys)):
            col_name = f"quality_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}"
            quality_score_cols.append(col_name)
            df[col_name] = df['quality'].apply(lambda x: x.get(key, {}).get('score') if isinstance(x, dict) and isinstance(x.get(key), dict) else (x.get(key) if isinstance(x, dict) and isinstance(x.get(key), (int, float)) else pd.NA))
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        print(f"Created {len(feature_score_cols)} feature score columns and {len(quality_score_cols)} quality score columns.")
        # --- End data preparation for statistical analysis ---

        # Features and Quality analysis (Original EDA loop - can be simplified or removed if stats are done below)
        # For now, let's keep it as it populates eda_results for the summary CSV.
        # The new statistical analysis will be more detailed.
        for category, col_name in [("core_features", "features"), ("argument_quality", "quality")]:
            all_item_stats = defaultdict(lambda: {'count': 0, 'scores': []})
            total_comments_with_category = 0
            
            for idx, row in df.iterrows():
                items_dict = row[col_name]
                if isinstance(items_dict, dict) and items_dict:
                    total_comments_with_category +=1
                    for item_name, score in items_dict.items():
                        all_item_stats[item_name]['count'] += 1
                        if isinstance(score, dict) and 'score' in score and isinstance(score['score'], (int, float)):
                            all_item_stats[item_name]['scores'].append(score['score'])
                        elif isinstance(score, (int, float)):
                             all_item_stats[item_name]['scores'].append(score)
                        else:
                            print(f"Warning: Non-numeric or unparseable score '{score}' for item '{item_name}' in comment {row.get('file_id', idx)}. Skipping score.")


            category_summary = {}
            for item_name, item_stat_values in all_item_stats.items():
                scores_arr = np.array(item_stat_values['scores'])
                if len(scores_arr) > 0:
                    category_summary[item_name] = {
                        'mention_frequency_in_comments_with_category_data': item_stat_values['count'] / total_comments_with_category if total_comments_with_category > 0 else 0,
                        'overall_mention_count': item_stat_values['count'],
                        'average_score': np.mean(scores_arr) if len(scores_arr) > 0 else None,
                        'median_score': np.median(scores_arr) if len(scores_arr) > 0 else None,
                        'std_dev_score': np.std(scores_arr) if len(scores_arr) > 0 else None,
                        'min_score': np.min(scores_arr) if len(scores_arr) > 0 else None,
                        'max_score': np.max(scores_arr) if len(scores_arr) > 0 else None,
                    }
                else:
                     category_summary[item_name] = {
                        'mention_frequency_in_comments_with_category_data': item_stat_values['count'] / total_comments_with_category if total_comments_with_category > 0 else 0,
                        'overall_mention_count': item_stat_values['count'],
                        'average_score': None, 'median_score': None, 'std_dev_score': None,
                        'min_score': None, 'max_score': None,
                    }
            eda_results[category] = category_summary
        
        print("EDA completed.")

        # --- 3. 产出CSV 1: EDA统计摘要 ---
        print(f"Saving EDA summary to {eda_summary_csv_path}...")
        # Flatten EDA results for CSV
        eda_df_list = []
        if 'stance_distribution' in eda_results and eda_results['stance_distribution']:
            for stance, freq in eda_results['stance_distribution'].items():
                eda_df_list.append({'category': 'stance_distribution', 'item': stance, 'metric': 'frequency', 'value': freq})

        if 'source_type_distribution' in eda_results and eda_results['source_type_distribution']:
            for source_type_val, freq in eda_results['source_type_distribution'].items():
                eda_df_list.append({'category': 'source_type_distribution', 'item': source_type_val, 'metric': 'frequency', 'value': freq})

        # --- Add persona_id distribution to EDA CSV (NEW) ---
        if 'persona_id_distribution' in eda_results and eda_results['persona_id_distribution']:
            for persona_val, freq in eda_results['persona_id_distribution'].items():
                eda_df_list.append({'category': 'persona_id_distribution', 'item': persona_val, 'metric': 'frequency', 'value': freq})
        if 'stance_distribution_by_persona' in eda_results and eda_results['stance_distribution_by_persona']:
            for persona, stance_dist in eda_results['stance_distribution_by_persona'].items():
                for stance_label, freq in stance_dist.items():
                     eda_df_list.append({'category': 'stance_distribution_by_persona', 'item': f'{persona}_{stance_label}', 'metric': 'frequency', 'value': freq})
        # --- End adding persona_id distribution ---

        for category_name, items in eda_results.items():
            if category_name in ['stance_distribution', 'source_type_distribution', 'persona_id_distribution', 'stance_distribution_by_persona']: # Updated condition to include new items
                continue
            if isinstance(items, dict):
                for item_name, metrics in items.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            eda_df_list.append({'category': category_name, 'item': item_name, 'metric': metric_name, 'value': value})
        
        if eda_df_list:
            pd.DataFrame(eda_df_list).to_csv(eda_summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"EDA summary saved.")
        else:
            print("No EDA results to save.")

        # --- START: Detailed Statistical Analysis (Checklist Item 3) ---
        statistical_results = []
        all_score_columns_to_analyze = feature_score_cols + quality_score_cols

        # Get unique persona_ids for iteration, excluding any NaN if they exist
        personas = df['persona_id'].dropna().unique().tolist()
        human_expert_data = df[df['persona_id'] == 'human_expert']
        gemini_all_data = df[df['persona_id'].str.startswith('gemini', na=False)] # All gemini personas + baseline

        # For effect sizes, define helper functions or use a library if available
        # Basic Cohen's d for two independent groups
        def cohen_d(x, y):
            nx = len(x)
            ny = len(y)
            dof = nx + ny - 2
            return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

        # --- START: New Debug Prints ---
        print("\n--- Debug: Pre-Statistical Analysis Data Check ---")
        print("Human Expert Data (.head()):")
        print(human_expert_data.head())
        print("\nHuman Expert Data (.columns):")
        print(human_expert_data.columns)
        print("\nHuman Expert Data (.info()):")
        human_expert_data.info(verbose=True, show_counts=True)
        
        print("\nGemini Data (All Personas) (.head()):")
        print(gemini_all_data.head())
        print("\nGemini Data (All Personas) (.columns):")
        print(gemini_all_data.columns)
        print("\nGemini Data (All Personas) (.info()):")
        gemini_all_data.info(verbose=True, show_counts=True)

        print("\nall_score_columns_to_analyze:")
        print(all_score_columns_to_analyze)

        if 'feature_Symbolism' in human_expert_data.columns:
            print(f"\nNon-NaN count for 'feature_Symbolism' in human_expert_data: {human_expert_data['feature_Symbolism'].notna().sum()} out of {len(human_expert_data)}")
        else:
            print("\n'feature_Symbolism' not found in human_expert_data columns.")
            
        if 'feature_Symbolism' in gemini_all_data.columns:
            print(f"Non-NaN count for 'feature_Symbolism' in gemini_all_data: {gemini_all_data['feature_Symbolism'].notna().sum()} out of {len(gemini_all_data)}")
        else:
            print("\n'feature_Symbolism' not found in gemini_all_data columns.")
        print("--- End: Pre-Statistical Analysis Data Check ---\n")
        # --- END: New Debug Prints ---

        for score_col in all_score_columns_to_analyze:
            print(f"\nAnalyzing column: {score_col}")
            # 1. Human Experts vs Gemini (All)
            group1_human = human_expert_data[score_col].dropna()
            group2_gemini_all = gemini_all_data[score_col].dropna()

            # --- Debug Prints Start ---
            print(f"  Debug: Human group for {score_col}: shape={group1_human.shape}, len={len(group1_human)}, NaNs_in_original_series={human_expert_data[score_col].isna().sum()}")
            print(f"  Debug: Gemini (All) group for {score_col}: shape={group2_gemini_all.shape}, len={len(group2_gemini_all)}, NaNs_in_original_series={gemini_all_data[score_col].isna().sum()}")
            # --- Debug Prints End ---
            
            desc_stats_human = group1_human.agg(['mean', 'median', 'std']).to_dict()
            desc_stats_gemini_all = group2_gemini_all.agg(['mean', 'median', 'std']).to_dict()

            if len(group1_human) > 1 and len(group2_gemini_all) > 1:
                # Shapiro-Wilk for normality
                norm_human_stat, norm_human_p = stats.shapiro(group1_human)
                norm_gemini_all_stat, norm_gemini_all_p = stats.shapiro(group2_gemini_all)

                if norm_human_p > 0.05 and norm_gemini_all_p > 0.05: # Both normal
                    levene_stat, levene_p = stats.levene(group1_human, group2_gemini_all)
                    if levene_p > 0.05: # Equal variances
                        t_stat, p_val = stats.ttest_ind(group1_human, group2_gemini_all, equal_var=True)
                        test_used = "t-test (equal var)"
                    else: # Unequal variances
                        t_stat, p_val = stats.ttest_ind(group1_human, group2_gemini_all, equal_var=False)
                        test_used = "t-test (unequal var)"
                    eff_size = cohen_d(group1_human, group2_gemini_all) if len(group1_human) > 0 and len(group2_gemini_all) > 0 else np.nan
                else: # At least one not normal
                    u_stat, p_val = stats.mannwhitneyu(group1_human, group2_gemini_all, alternative='two-sided')
                    test_used = "Mann-Whitney U"
                    # Effect size for Mann-Whitney U: r = Z / sqrt(N)
                    # Z can be approximated from U. Or use a library. For simplicity, not calculating here for now.
                    # eff_size = (u_stat - (len(group1_human)*len(group2_gemini_all)/2)) / np.sqrt((len(group1_human)*len(group2_gemini_all)*(len(group1_human)+len(group2_gemini_all)+1))/12)
                    eff_size = np.nan # Placeholder for non-parametric effect size
                
                statistical_results.append({
                    'comparison_type': 'Human vs Gemini (All)',
                    'metric': score_col,
                    'human_mean': desc_stats_human.get('mean'), 'human_median': desc_stats_human.get('median'), 'human_std': desc_stats_human.get('std'),
                    'gemini_all_mean': desc_stats_gemini_all.get('mean'), 'gemini_all_median': desc_stats_gemini_all.get('median'), 'gemini_all_std': desc_stats_gemini_all.get('std'),
                    'test_used': test_used, 'statistic': t_stat if 't-test' in test_used else u_stat, 'p_value': p_val, 'effect_size': eff_size
                })

            # 2. Human Experts vs. Each Gemini Persona
            gemini_personas = [p for p in personas if p.startswith('gemini') and p != 'gemini_baseline' and p != 'gemini_unknown_persona' and p != 'gemini_parse_error']
            if 'gemini_baseline' in personas:
                 gemini_personas.insert(0, 'gemini_baseline') # Add baseline to iterate over as well

            for persona in gemini_personas:
                group2_persona = df[(df['persona_id'] == persona)][score_col].dropna()
                desc_stats_persona = group2_persona.agg(['mean', 'median', 'std']).to_dict()

                # --- Debug Prints Start ---
                print(f"    Debug: Human group for {score_col} vs {persona}: shape={group1_human.shape}, len={len(group1_human)}, NaNs_in_original_series_human={human_expert_data[score_col].isna().sum()}")
                print(f"    Debug: {persona} group for {score_col}: shape={group2_persona.shape}, len={len(group2_persona)}, NaNs_in_original_series_persona={df[(df['persona_id'] == persona)][score_col].isna().sum()}")
                # --- Debug Prints End ---

                if len(group1_human) > 1 and len(group2_persona) > 1:
                    # Normality and test selection (similar to above)
                    norm_p2_stat, norm_p2_p = stats.shapiro(group2_persona)
                    if norm_human_p > 0.05 and norm_p2_p > 0.05:
                        levene_stat, levene_p = stats.levene(group1_human, group2_persona)
                        t_stat, p_val = stats.ttest_ind(group1_human, group2_persona, equal_var=(levene_p > 0.05))
                        test_used_pvp = "t-test"
                        eff_size_pvp = cohen_d(group1_human, group2_persona) if len(group1_human) > 0 and len(group2_persona) > 0 else np.nan
                    else:
                        u_stat, p_val = stats.mannwhitneyu(group1_human, group2_persona, alternative='two-sided')
                        test_used_pvp = "Mann-Whitney U"
                        eff_size_pvp = np.nan # Placeholder
                    
                    statistical_results.append({
                        'comparison_type': f'Human vs {persona}',
                        'metric': score_col,
                        'human_mean': desc_stats_human.get('mean'), 'human_median': desc_stats_human.get('median'), 'human_std': desc_stats_human.get('std'),
                        f'{persona}_mean': desc_stats_persona.get('mean'), f'{persona}_median': desc_stats_persona.get('median'), f'{persona}_std': desc_stats_persona.get('std'),
                        'test_used': test_used_pvp, 'statistic': t_stat if 't-test' in test_used_pvp else u_stat, 'p_value': p_val, 'effect_size': eff_size_pvp
                    })
            
            # 3. ANOVA/Kruskal-Wallis for all Gemini personas (and optionally human_expert as one group)
            # This requires more careful handling of group collection and post-hoc tests
            # For simplicity in this step, we will skip the multi-group comparison (ANOVA/K-W) here
            # but it can be added. The plan mentions it. Focus for now is on pairwise.

        # Save statistical_results to a CSV
        if statistical_results:
            stats_df = pd.DataFrame(statistical_results)
            stats_output_path = output_analysis_dir / "feature_quality_statistical_comparison.csv"
            stats_df.to_csv(stats_output_path, index=False, encoding='utf-8-sig')
            print(f"Detailed statistical comparison saved to {stats_output_path}")
        else:
            print("No statistical results generated.")
        # --- END: Detailed Statistical Analysis ---

        # --- 4. 特征向量构建 for t-SNE/UMAP ---
        print("Constructing feature vectors for dimensionality reduction...")
        # Note: The original feature vector construction used the dicts 'features' and 'quality'
        # For consistency with the above statistical analysis, we could use the newly created feature_ and quality_ columns
        # Or, keep the original method. Let's keep original for now to not break t-SNE if column names differ slightly.
        # However, it might be better to use the flattened columns. For now, no change here.
        all_feature_keys_tsne = set() # Renaming to avoid clash
        for col_name_tsne in ['features', 'quality']: # Renaming to avoid clash
            if col_name_tsne in df.columns:
                for items_dict in df[col_name_tsne]:
                    if isinstance(items_dict, dict):
                        all_feature_keys_tsne.update(items_dict.keys())
        
        feature_vectors = []
        if not all_feature_keys_tsne:
            print("Warning: No features found to build vectors for dimensionality reduction (using original dict method). Skipping t-SNE/UMAP.")
        else:
            sorted_feature_keys_tsne = sorted(list(all_feature_keys_tsne))
            for idx, row in df.iterrows():
                vector = []
                # Ensure 'features' and 'quality' exist and are dicts
                row_features = row.get('features', {})
                if not isinstance(row_features, dict): row_features = {}
                row_quality = row.get('quality', {})
                if not isinstance(row_quality, dict): row_quality = {}
                
                current_comment_items = {**row_features, **row_quality} # Combine feature and quality dicts
                
                for key in sorted_feature_keys_tsne:
                    item_value = current_comment_items.get(key)
                    if isinstance(item_value, dict) and 'score' in item_value and isinstance(item_value['score'], (int, float)):
                        vector.append(item_value['score'])
                    elif isinstance(item_value, (int, float)):
                        vector.append(item_value)
                    else:
                        vector.append(0.0) # Default if key not found, or value is not a direct score or expected dict
                feature_vectors.append(vector)
            
            feature_vectors_np = np.array(feature_vectors)
            print(f"Feature vectors constructed. Shape: {feature_vectors_np.shape}")

            # Prepare labels for coloring plots (use stance_label_en if available, or source_type)
            labels_for_plot = None
            hue_column_name = None
            style_column_name = None # For potentially styling by source_type if coloring by persona_id

            # --- Modify plot labeling to prioritize persona_id ---
            if 'persona_id' in df.columns:
                labels_for_plot = df['persona_id'].fillna('Unknown').astype(str)
                hue_column_name = 'Persona ID'
                if 'source_type' in df.columns:
                     # If we color by persona, we can style by source_type (e.g. human vs gemini)
                     style_column_name = 'source_type' 
            elif 'stance_label_en' in df.columns:
                labels_for_plot = df['stance_label_en'].fillna('Unknown').astype(str)
                hue_column_name = 'Stance'
                if 'source_type' in df.columns:
                    style_column_name = 'source_type'
            elif 'source_type' in df.columns: # Fallback to source_type if stance is not available
                labels_for_plot = df['source_type'].fillna('Unknown').astype(str)
                hue_column_name = 'Source Type'
            # --- End plot labeling modification ---
            
            dim_reduction_coords_data = {'file_id': df['file_id'] if 'file_id' in df.columns else range(len(df))}
            # Add source_type to the coordinates data for potential use in external tools
            if 'source_type' in df.columns:
                dim_reduction_coords_data['source_type'] = df['source_type']
            # --- Add persona_id to coordinates data (NEW) ---
            if 'persona_id' in df.columns:
                dim_reduction_coords_data['persona_id'] = df['persona_id']
            # --- End adding persona_id ---

            # --- 5 & 6. t-SNE可视化与保存 ---
            if feature_vectors_np.shape[0] > 0 and feature_vectors_np.shape[1] > 0 : # Ensure non-empty feature vectors
                print("Performing t-SNE...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, feature_vectors_np.shape[0]-1) if feature_vectors_np.shape[0] > 1 else 1) # Adjust perplexity
                tsne_results = tsne.fit_transform(feature_vectors_np)
                
                dim_reduction_coords_data['tsne_x'] = tsne_results[:,0]
                dim_reduction_coords_data['tsne_y'] = tsne_results[:,1]
                # Add to main df
                if 'tsne_x' not in df.columns:
                    df['tsne_x'] = pd.NA
                if 'tsne_y' not in df.columns:
                    df['tsne_y'] = pd.NA
                df.loc[df.index[:len(tsne_results)], 'tsne_x'] = tsne_results[:,0]
                df.loc[df.index[:len(tsne_results)], 'tsne_y'] = tsne_results[:,1]

                plt.figure(figsize=(14, 10)) # Increased figure size
                if labels_for_plot is not None and len(labels_for_plot) == len(tsne_results):
                    current_style_column = df.get(style_column_name, None) if style_column_name else None
                    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels_for_plot, style=current_style_column, palette="viridis", legend="full")
                    plt.title(f't-SNE visualization of Critique Features (colored by {hue_column_name}{f", styled by {style_column_name}" if style_column_name else ""})')
                else:
                    current_style_column = df.get(style_column_name, None) if style_column_name else None
                    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], style=current_style_column, palette="viridis")
                    plt.title(f't-SNE visualization of Critique Features{f" (styled by {style_column_name})" if style_column_name else ""}')
                
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.savefig(tsne_plot_path)
                plt.close()
                print(f"t-SNE plot saved to {tsne_plot_path}")

                # --- START: Save t-SNE data to CSV (NEW from PLAN - Checklist 2.2, 2.3) ---
                tsne_data_for_csv = pd.DataFrame({
                    'file_id': df['file_id'].iloc[:len(tsne_results)] if 'file_id' in df.columns else range(len(tsne_results)),
                    'persona_id': labels_for_plot.iloc[:len(tsne_results)] if labels_for_plot is not None and hasattr(labels_for_plot, 'iloc') else pd.NA,
                    'source_type': df['source_type'].iloc[:len(tsne_results)] if 'source_type' in df.columns else pd.NA,
                    'tsne_x': tsne_results[:,0],
                    'tsne_y': tsne_results[:,1]
                })
                tsne_csv_path = output_eda_plots_dir / "features_tsne_visualization_human_gemini_data.csv"
                tsne_data_for_csv.to_csv(tsne_csv_path, index=False, encoding='utf-8-sig')
                print(f"t-SNE data saved to {tsne_csv_path}")
                # --- END: Save t-SNE data to CSV ---

            else:
                print("Skipping t-SNE due to empty or invalid feature vectors.")

            # --- 7 & 8. UMAP可视化与保存 ---
            if UMAP_AVAILABLE and feature_vectors_np.shape[0] > 0 and feature_vectors_np.shape[1] > 0:
                print("Performing UMAP...")
                try:
                    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, feature_vectors_np.shape[0]-1) if feature_vectors_np.shape[0] > 1 else 1, min_dist=0.1)
                    umap_results = umap_reducer.fit_transform(feature_vectors_np)

                    dim_reduction_coords_data['umap_x'] = umap_results[:,0]
                    dim_reduction_coords_data['umap_y'] = umap_results[:,1]
                    # Add to main df
                    if 'umap_x' not in df.columns:
                        df['umap_x'] = pd.NA
                    if 'umap_y' not in df.columns:
                        df['umap_y'] = pd.NA
                    df.loc[df.index[:len(umap_results)], 'umap_x'] = umap_results[:,0]
                    df.loc[df.index[:len(umap_results)], 'umap_y'] = umap_results[:,1]

                    plt.figure(figsize=(14, 10)) # Increased figure size
                    if labels_for_plot is not None and len(labels_for_plot) == len(umap_results):
                        current_style_column = df.get(style_column_name, None) if style_column_name else None
                        sns.scatterplot(x=umap_results[:,0], y=umap_results[:,1], hue=labels_for_plot, style=current_style_column, palette="viridis", legend="full")
                        plt.title(f'UMAP visualization of Critique Features (colored by {hue_column_name}{f", styled by {style_column_name}" if style_column_name else ""})')
                    else:
                        current_style_column = df.get(style_column_name, None) if style_column_name else None
                        sns.scatterplot(x=umap_results[:,0], y=umap_results[:,1], style=current_style_column, palette="viridis")
                        plt.title(f'UMAP visualization of Critique Features{f" (styled by {style_column_name})" if style_column_name else ""}')
                    plt.xlabel('UMAP Dimension 1')
                    plt.ylabel('UMAP Dimension 2')
                    plt.savefig(umap_plot_path)
                    plt.close()
                    print(f"UMAP plot saved to {umap_plot_path}")

                    # --- START: Save UMAP data to CSV (NEW from PLAN - Checklist 3.2, 3.3) ---
                    umap_data_for_csv = pd.DataFrame({
                        'file_id': df['file_id'].iloc[:len(umap_results)] if 'file_id' in df.columns else range(len(umap_results)),
                        'persona_id': labels_for_plot.iloc[:len(umap_results)] if labels_for_plot is not None and hasattr(labels_for_plot, 'iloc') else pd.NA,
                        'source_type': df['source_type'].iloc[:len(umap_results)] if 'source_type' in df.columns else pd.NA,
                        'umap_x': umap_results[:,0],
                        'umap_y': umap_results[:,1]
                    })
                    umap_csv_path = output_eda_plots_dir / "features_umap_visualization_human_gemini_data.csv"
                    umap_data_for_csv.to_csv(umap_csv_path, index=False, encoding='utf-8-sig')
                    print(f"UMAP data saved to {umap_csv_path}")
                    # --- END: Save UMAP data to CSV ---

                except Exception as e:
                    print(f"Error during UMAP processing: {e}. Skipping UMAP.")
            elif not UMAP_AVAILABLE:
                print("UMAP visualization skipped as umap-learn package is not available.")
            else:
                print("Skipping UMAP due to empty or invalid feature vectors.")
            
            # --- 9. 产出CSV 2: 降维坐标 ---
            print(f"Saving dimensionality reduction coordinates to {dim_reduction_coords_csv_path}...")
            pd.DataFrame(dim_reduction_coords_data).to_csv(dim_reduction_coords_csv_path, index=False, encoding='utf-8-sig')
            print("Dimensionality reduction coordinates saved.")

        # --- START: Calculate and Save Centroids (Checklist Item 5) ---
        if 'tsne_x' in df.columns and 'persona_id' in df.columns: # Check if t-SNE was run and persona_id exists
            tsne_centroids = df.groupby('persona_id')[['tsne_x', 'tsne_y']].mean().reset_index()
            tsne_centroids.rename(columns={'tsne_x': 'mean_tsne_x', 'tsne_y': 'mean_tsne_y'}, inplace=True)
            print("\nt-SNE Centroids by Persona ID:")
            print(tsne_centroids)
        else:
            tsne_centroids = pd.DataFrame() # Empty dataframe if no t-SNE data

        if 'umap_x' in df.columns and 'persona_id' in df.columns: # Check if UMAP was run
            umap_centroids = df.groupby('persona_id')[['umap_x', 'umap_y']].mean().reset_index()
            umap_centroids.rename(columns={'umap_x': 'mean_umap_x', 'umap_y': 'mean_umap_y'}, inplace=True)
            print("\nUMAP Centroids by Persona ID:")
            print(umap_centroids)
        else:
            umap_centroids = pd.DataFrame() # Empty dataframe if no UMAP data

        # Merge centroid dataframes
        if not tsne_centroids.empty and not umap_centroids.empty:
            all_centroids = pd.merge(tsne_centroids, umap_centroids, on='persona_id', how='outer')
        elif not tsne_centroids.empty:
            all_centroids = tsne_centroids
        elif not umap_centroids.empty:
            all_centroids = umap_centroids
        else:
            all_centroids = pd.DataFrame() # Empty if neither was computed

        if not all_centroids.empty:
            centroids_output_path = output_analysis_dir / "feature_space_centroids.csv"
            all_centroids.to_csv(centroids_output_path, index=False, encoding='utf-8-sig')
            print(f"\nFeature space centroids saved to {centroids_output_path}")
        else:
            print("\nNo centroid data generated or saved.")
        # --- END: Calculate and Save Centroids ---

        print("Script part 1 finished.")
        # --- END OF ORIGINAL main() CODE ---

    except Exception as e:
        # Checklist 3.6: Optional error handling to log critical errors
        # Restore stderr first to ensure the error message itself goes to the console (if possible)
        sys.stderr = original_stderr 
        print(f"CRITICAL SCRIPT ERROR: {e}\nTraceback:\n{traceback.format_exc()}", file=sys.stderr) # To original stderr (console)
        if log_file_handle and not log_file_handle.closed:
             print(f"CRITICAL SCRIPT ERROR: {e}\nTraceback:\n{traceback.format_exc()}", file=log_file_handle) # Also to log file
        # raise # Uncomment if script should exit with non-zero code on error
    finally:
        # Ensure stdout and stderr are restored even if an error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()
        # Checklist 3.8: Confirmation message to original stdout
        print(f"Debug log updated at: {debug_log_path}") 

if __name__ == '__main__':
    main() 