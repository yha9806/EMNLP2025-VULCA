import os
import glob
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Please update these paths according to your project structure and data location.
# Base Path to the directory containing subdirectories of human expert .txt files.
# The script will search for .txt files recursively in all subdirectories of this base path.
# Example: BASE_DATA_DIRECTORY = "/path/to/your/data/text/processed/"
# Relative path from script location (experiment/human_expert/src/) to the data directory
# This path will be resolved from the Current Working Directory when the script is run.
# Assuming CWD is '.../experiment/', this points to '.../data/text/processed/'
# --- MODIFIED: Path relative to workspace root --- #
BASE_DATA_DIRECTORY = "data/text/processed/" 

# Path for the consolidated master output CSV file.
OUTPUT_CSV_FILE_MASTER = "experiment/human_expert/result/human_expert_features_consolidated.csv"

# Directory for per-scholar/work specific CSV files.
PER_SCHOLAR_OUTPUT_DIR = "experiment/human_expert/result/per_scholar_results/"

# --- Helper Function: Load Text ---
def load_critique_from_txt(file_path):
    """Loads text content from a given file path with UTF-8 encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Phase 1: Human Expert Comment Text Benchmark Feature Extraction...")

    # Create output directories if they don't exist
    if PER_SCHOLAR_OUTPUT_DIR and not os.path.exists(PER_SCHOLAR_OUTPUT_DIR):
        os.makedirs(PER_SCHOLAR_OUTPUT_DIR)
        print(f"Created per-scholar output directory: {PER_SCHOLAR_OUTPUT_DIR}")
    
    master_output_dir = os.path.dirname(OUTPUT_CSV_FILE_MASTER)
    if master_output_dir and not os.path.exists(master_output_dir):
        os.makedirs(master_output_dir)
        print(f"Created master output directory: {master_output_dir}")

    # Initialize the zero-shot classification pipeline
    # Uses GPU if available, otherwise CPU
    print("Loading zero-shot classification model...")
    try:
        analyzer = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Please ensure you have an internet connection for the first download and that PyTorch is installed correctly.")
        exit()

    # Define candidate labels as per the README, now with English as primary and Chinese in comments
    # English labels for the model
    stance_labels_en = [
        "Historical Research", "Aesthetic Appreciation", "Socio-cultural Interpretation",
        "Comparative Analysis", "Theoretical Construction", "Critical Inquiry",
        "High Praise", "Objective Description", "Mild Criticism", "Strong Negation"
    ]
    feature_labels_en = [
        "Use of Color", "Brushwork Technique", "Texture Strokes (Chunfa)", "Line Quality", "Ink Application",
        "Layout and Structure", "Spatial Representation",
        "Artistic Conception", "Emotional Expression",
        "Subject Matter", "Genre", "Symbolism", "Historical Context", "Artist Biography",
        "Style/School", "Technique Inheritance & Innovation", "Cross-cultural Influence"
    ]
    quality_labels_en = [
        "Profound Insight", "Strong Argumentation", "Clear Logic", "Detailed Analysis",
        "Classical Citations", "Objective Viewpoint", "Superficial Treatment", "Overly General Content",
        "Lacks Examples", "Logical Gaps", "Subjective/Biased View"
    ]

    # Bilingual mappings (English to Chinese)
    stance_label_map = {
        "Historical Research": "历史考证型",
        "Aesthetic Appreciation": "美学鉴赏型",
        "Socio-cultural Interpretation": "社会文化解读型",
        "Comparative Analysis": "比较分析型",
        "Theoretical Construction": "理论建构型",
        "Critical Inquiry": "质疑与思辨型",
        "High Praise": "高度赞扬与推崇",
        "Objective Description": "客观中性描述",
        "Mild Criticism": "温和批评与保留",
        "Strong Negation": "强烈否定与驳斥"
    }
    feature_label_map = {
        "Use of Color": "色彩运用",
        "Brushwork Technique": "笔法技巧",
        "Texture Strokes (Chunfa)": "皴法特点",
        "Line Quality": "线条质量",
        "Ink Application": "墨法变化",
        "Layout and Structure": "布局与结构",
        "Spatial Representation": "空间营造",
        "Artistic Conception": "意境表达",
        "Emotional Expression": "情感传递",
        "Subject Matter": "主题内容",
        "Genre": "题材选择",
        "Symbolism": "象征意义",
        "Historical Context": "历史背景",
        "Artist Biography": "画家生平",
        "Style/School": "风格流派",
        "Technique Inheritance & Innovation": "技法传承与创新",
        "Cross-cultural Influence": "跨文化影响"
    }
    quality_label_map = {
        "Profound Insight": "见解深刻独到",
        "Strong Argumentation": "论证充分有力",
        "Clear Logic": "逻辑清晰严密",
        "Detailed Analysis": "细节分析具体",
        "Classical Citations": "引用经典佐证",
        "Objective Viewpoint": "观点客观公允",
        "Superficial Treatment": "论述流于表面",
        "Overly General Content": "内容较为宽泛",
        "Lacks Examples": "缺乏具体例证",
        "Logical Gaps": "逻辑存在跳跃",
        "Subjective/Biased View": "观点主观片面"
    }

    # This list is no longer needed as we save incrementally
    # all_critique_features = [] 
    
    # Find all .txt files in the base data directory and its subdirectories
    print(f"Searching for critique files recursively in: {os.path.abspath(BASE_DATA_DIRECTORY)}...")
    
    critique_files_found = []
    if not os.path.isdir(BASE_DATA_DIRECTORY):
        print(f"Error: BASE_DATA_DIRECTORY '{os.path.abspath(BASE_DATA_DIRECTORY)}' does not exist or is not a directory.")
        print("Please check the BASE_DATA_DIRECTORY path in the script.")
        exit()

    for root_dir, _, files in os.walk(BASE_DATA_DIRECTORY):
        for file in files:
            if file.endswith(".txt"):
                critique_files_found.append(os.path.join(root_dir, file))

    if not critique_files_found:
        print(f"No .txt files found recursively in '{os.path.abspath(BASE_DATA_DIRECTORY)}'.")
        print("Please ensure your .txt files are organized within subdirectories of this path.")
    else:
        print(f"Found {len(critique_files_found)} critique file(s) to process.")

    # Initialize a set to keep track of files for which headers have been written
    # This is for both master and per-scholar files
    headers_written_files = set()

    for file_path in tqdm(critique_files_found, desc="Processing critiques"):
        # Create a file_id that is relative to the BASE_DATA_DIRECTORY
        # This helps in identifying the source, e.g., "Author_Work/Chapter.txt"
        relative_file_path = os.path.relpath(file_path, BASE_DATA_DIRECTORY)
        # tqdm.write(f"Processing: {relative_file_path}...") # tqdm.write is better for progress bars
        
        critique_text = load_critique_from_txt(file_path)

        if critique_text:
            extracted_stance_data = {}
            extracted_features_data = {}
            extracted_quality_data = {}

            try:
                # 1. Extract Evaluative Stance Spectrum
                stance_results = analyzer(critique_text, candidate_labels=stance_labels_en, multi_label=False)
                en_stance_label = stance_results['labels'][0]
                extracted_stance_data = {
                    "label_en": en_stance_label,
                    "label_zh": stance_label_map.get(en_stance_label, ""), 
                    "score": stance_results['scores'][0]
                }

                # 2. Extract Core Focal Points
                feature_results = analyzer(critique_text, candidate_labels=feature_labels_en, multi_label=True)
                for label, score in zip(feature_results['labels'], feature_results['scores']):
                    extracted_features_data[label] = {
                        "score": score,
                        "zh": feature_label_map.get(label, "") 
                    }
                
                # 3. Extract Argumentative Quality Features
                quality_results = analyzer(critique_text, candidate_labels=quality_labels_en, multi_label=True)
                for label, score in zip(quality_results['labels'], quality_results['scores']):
                    extracted_quality_data[label] = {
                        "score": score,
                        "zh": quality_label_map.get(label, "") 
                    }

                current_critique_data = {
                    "file_id": relative_file_path,
                    "text_preview": critique_text[:200] + "..." if len(critique_text) > 200 else critique_text, 
                    "stance_label_en": extracted_stance_data.get('label_en'),
                    "stance_label_zh": extracted_stance_data.get('label_zh'),
                    "stance_score": extracted_stance_data.get('score'),
                    "features": str(extracted_features_data), # Convert dict to string for CSV storage
                    "quality": str(extracted_quality_data)    # Convert dict to string for CSV storage
                }
                
                # Convert single data point to DataFrame for consistent saving
                df_single_row = pd.DataFrame([current_critique_data])

                # --- Save to Consolidated Master CSV ---
                master_file_exists = os.path.isfile(OUTPUT_CSV_FILE_MASTER)
                write_header_master = not master_file_exists or OUTPUT_CSV_FILE_MASTER not in headers_written_files
                
                df_single_row.to_csv(OUTPUT_CSV_FILE_MASTER, mode='a', header=write_header_master, index=False, encoding='utf-8-sig')
                if write_header_master:
                    headers_written_files.add(OUTPUT_CSV_FILE_MASTER)

                # --- Save to Per-Scholar/Work CSV ---
                # Determine scholar/work identifier (first part of relative_file_path)
                scholar_identifier = os.path.normpath(relative_file_path).split(os.sep)[0]
                # Sanitize scholar_identifier to be a valid filename (optional, but good practice)
                # For now, assuming it's clean enough or user ensures valid directory names
                
                if scholar_identifier and scholar_identifier != ".": # Ensure it's a valid subdir
                    scholar_csv_path = os.path.join(PER_SCHOLAR_OUTPUT_DIR, scholar_identifier + ".csv")
                    scholar_file_exists = os.path.isfile(scholar_csv_path)
                    write_header_scholar = not scholar_file_exists or scholar_csv_path not in headers_written_files

                    df_single_row.to_csv(scholar_csv_path, mode='a', header=write_header_scholar, index=False, encoding='utf-8-sig')
                    if write_header_scholar:
                         headers_written_files.add(scholar_csv_path)
                
                # tqdm.write(f"Successfully processed and saved: {relative_file_path}")

            except Exception as e:
                tqdm.write(f"Error during NLP analysis for file {relative_file_path}: {e}")
        else:
            tqdm.write(f"Skipping file {relative_file_path} due to loading error.")
        # tqdm.write("-" * 30) # Optional separator, can make log noisy with tqdm

    # Final messages after loop
    if critique_files_found:
        print(f"\nProcessing complete.")
        print(f"Consolidated results saved to: {os.path.abspath(OUTPUT_CSV_FILE_MASTER)}")
        print(f"Per-scholar results saved in directory: {os.path.abspath(PER_SCHOLAR_OUTPUT_DIR)}")
    else:
        print("No files were processed. No CSV outputs created.")

    print("Script finished.") 