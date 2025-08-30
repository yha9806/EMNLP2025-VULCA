import pandas as pd
import json
import re
from pathlib import Path
from collections import Counter
import ast
import numpy as np

# --- file_id 解析函数 ---
def extract_scholar_from_file_id(file_id_str: str):
    if not isinstance(file_id_str, str):
        return "Unknown_Scholar"
    # 尝试从 "学者名（英文名）_书名\文件名" 或 "学者名_书名\文件名" 中提取
    # Windows路径分隔符是 \, 在字符串中需要写为 \\
    # 如果是Linux/Mac路径，分隔符是 /
    part_before_separator = file_id_str.split('\\')[0].split('/')[0]
    
    # 优先尝试匹配带括号的英文名模式: 匹配 "中文名（英文名）"
    match_with_en = re.match(r"([^（]+（[^）]+）)", part_before_separator)
    if match_with_en:
        return match_with_en.group(1)
    
    # 否则，尝试以下划线为界获取第一部分 (处理 "中文名_书名" 或 "英文名_书名" 的情况)
    # 确保不只是取到文件名的一部分（如果没有下划线且没有括号）
    if '_' in part_before_separator:
        return part_before_separator.split('_', 1)[0]
        
    # 如果都没有匹配，返回整个部分作为备用 (可能只有学者名或书名，没有下划线)
    return part_before_separator

def extract_book_id_from_file_id(file_id_str: str):
    if not isinstance(file_id_str, str):
        return "Unknown_Book"
    # Windows路径分隔符是 \, 在字符串中需要写为 \\
    # 如果是Linux/Mac路径，分隔符是 /
    return file_id_str.split('\\')[0].split('/')[0]

# --- 特征聚合函数 ---
def aggregate_features(df: pd.DataFrame, group_by_column: str, all_feature_labels: set, all_quality_labels: set) -> pd.DataFrame:
    print(f"Aggregating features grouped by: {group_by_column}...")
    aggregated_data = []
    
    grouped = df.groupby(group_by_column)

    for group_name, group_df in grouped:
        if group_name is None or pd.isna(group_name): # Skip if group_name is None or NaN
            print(f"Skipping group with None/NaN name in column {group_by_column}")
            continue

        current_group_aggregated_row = {group_by_column: group_name}
        
        current_group_aggregated_row['num_segments'] = len(group_df)
        
        # Stance聚合
        if not group_df['stance_label_en'].empty and group_df['stance_label_en'].notna().any():
            current_group_aggregated_row['mode_stance_label_en'] = group_df['stance_label_en'].mode().iloc[0] if not group_df['stance_label_en'].mode().empty else 'N/A'
        else:
            current_group_aggregated_row['mode_stance_label_en'] = 'N/A'
            
        if not group_df['stance_label_zh'].empty and group_df['stance_label_zh'].notna().any():
            current_group_aggregated_row['mode_stance_label_zh'] = group_df['stance_label_zh'].mode().iloc[0] if not group_df['stance_label_zh'].mode().empty else 'N/A'
        else:
             current_group_aggregated_row['mode_stance_label_zh'] = 'N/A'

        current_group_aggregated_row['avg_stance_score'] = group_df['stance_score'].mean()

        # Features 聚合
        feature_scores_collection = {label: [] for label in all_feature_labels}
        for features_dict_list in group_df['features'].dropna(): # Iterate over list of dicts
            if isinstance(features_dict_list, list): # Ensure it's a list
                 for features_dict in features_dict_list: # Iterate over dicts in the list
                    if isinstance(features_dict, dict):
                        for label, details in features_dict.items():
                            if label in feature_scores_collection and isinstance(details, dict) and 'score' in details:
                                feature_scores_collection[label].append(details['score'])
            elif isinstance(features_dict_list, dict): # Handle if it's a single dict not a list
                 features_dict = features_dict_list
                 if isinstance(features_dict, dict):
                    for label, details in features_dict.items():
                        if label in feature_scores_collection and isinstance(details, dict) and 'score' in details:
                            feature_scores_collection[label].append(details['score'])


        for label, scores in feature_scores_collection.items():
            col_name = f"avg_feat_{label.replace(' ', '_').replace('/', '_')}_score"
            current_group_aggregated_row[col_name] = np.mean(scores) if scores else np.nan

        # Quality 聚合
        quality_scores_collection = {label: [] for label in all_quality_labels}
        for quality_dict_list in group_df['quality'].dropna(): # Iterate over list of dicts
            if isinstance(quality_dict_list, list):  # Ensure it's a list
                for quality_dict in quality_dict_list: # Iterate over dicts in the list
                    if isinstance(quality_dict, dict):
                        for label, details in quality_dict.items():
                            if label in quality_scores_collection and isinstance(details, dict) and 'score' in details:
                                quality_scores_collection[label].append(details['score'])
            elif isinstance(quality_dict_list, dict): # Handle if it's a single dict not a list
                quality_dict = quality_dict_list
                if isinstance(quality_dict, dict):
                    for label, details in quality_dict.items():
                        if label in quality_scores_collection and isinstance(details, dict) and 'score' in details:
                            quality_scores_collection[label].append(details['score'])

        for label, scores in quality_scores_collection.items():
            col_name = f"avg_qual_{label.replace(' ', '_').replace('/', '_')}_score"
            current_group_aggregated_row[col_name] = np.mean(scores) if scores else np.nan
            
        aggregated_data.append(current_group_aggregated_row)
        
    return pd.DataFrame(aggregated_data)

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        # Handles empty strings, malformed strings, NaN, etc.
        return None # Or return {}, or [], depending on expected type for empty/error

def main():
    script_dir = Path(__file__).resolve().parent
    base_expert_dir = script_dir.parent # experiment/human_expert/
    
    # Input path
    input_csv_path = base_expert_dir / "result/human_expert_features_consolidated.csv"
    
    # Output paths
    output_dir = base_expert_dir / "result/analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_scholar_features_csv_path = output_dir / "aggregated_scholar_features.csv"
    output_book_features_csv_path = output_dir / "aggregated_book_features.csv"

    print(f"Loading data from: {input_csv_path}")
    if not input_csv_path.exists():
        print(f"ERROR: Input CSV file not found at {input_csv_path}")
        return
    df = pd.read_csv(input_csv_path)
    print(f"Successfully loaded {len(df)} rows.")

    # Preprocess file_id to create scholar_id and book_id
    print("Extracting scholar_id and book_id from file_id...")
    df['scholar_id'] = df['file_id'].apply(extract_scholar_from_file_id)
    df['book_id'] = df['file_id'].apply(extract_book_id_from_file_id)
    print(df[['file_id', 'scholar_id', 'book_id']].head())


    # Parse JSON-like string columns (features, quality)
    # ast.literal_eval is safer than json.loads for non-strict JSON
    print("Parsing 'features' and 'quality' columns...")
    # Ensure the columns are treated as strings before applying literal_eval
    df['features'] = df['features'].astype(str).apply(safe_literal_eval)
    df['quality'] = df['quality'].astype(str).apply(safe_literal_eval)

    # Discover all unique feature and quality labels from the dataset
    all_feature_labels = set()
    for item_list in df['features'].dropna():
        if isinstance(item_list, list): # Check if it's a list of dicts
            for item_dict in item_list:
                if isinstance(item_dict, dict):
                    all_feature_labels.update(item_dict.keys())
        elif isinstance(item_list, dict): # Check if it's a single dict
             all_feature_labels.update(item_list.keys())


    all_quality_labels = set()
    for item_list in df['quality'].dropna():
        if isinstance(item_list, list): # Check if it's a list of dicts
            for item_dict in item_list:
                if isinstance(item_dict, dict):
                    all_quality_labels.update(item_dict.keys())
        elif isinstance(item_list, dict): # Check if it's a single dict
            all_quality_labels.update(item_list.keys())
            
    print(f"Discovered {len(all_feature_labels)} unique feature labels.")
    print(f"Discovered {len(all_quality_labels)} unique quality labels.")


    # --- Aggregate by Scholar ---
    scholar_aggregated_df = aggregate_features(df, "scholar_id", all_feature_labels, all_quality_labels)
    if not scholar_aggregated_df.empty:
        scholar_aggregated_df.to_csv(output_scholar_features_csv_path, index=False, encoding='utf-8-sig')
        print(f"Aggregated scholar features saved to: {output_scholar_features_csv_path}")
    else:
        print("Scholar aggregation resulted in an empty DataFrame. No output file created.")

    # --- Aggregate by Book ---
    book_aggregated_df = aggregate_features(df, "book_id", all_feature_labels, all_quality_labels)
    if not book_aggregated_df.empty:
        book_aggregated_df.to_csv(output_book_features_csv_path, index=False, encoding='utf-8-sig')
        print(f"Aggregated book features saved to: {output_book_features_csv_path}")
    else:
        print("Book aggregation resulted in an empty DataFrame. No output file created.")

    print("Script aggregate_expert_features.py finished.")

if __name__ == '__main__':
    main() 