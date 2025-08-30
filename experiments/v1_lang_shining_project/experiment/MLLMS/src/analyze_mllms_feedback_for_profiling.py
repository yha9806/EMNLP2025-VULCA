import os
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# --- Helper Function: Load Text ---
def load_critique_from_txt(file_path):
    """Loads text content from a given file path with UTF-8 encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        tqdm.write(f"Error: File not found {file_path}")
        return None
    except Exception as e:
        tqdm.write(f"Error reading file {file_path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # MODIFIED: Path relative to project root
    BASE_DATA_DIRECTORY = "v1_lang_shining_project/experiment/MLLMS/feedbacks/"
    # Ensure this path is correct if running from a different CWD than project root.
    # For consistency, let's make it an absolute path based on a potential PROJECT_ROOT env var or relative to script.
    # However, for now, assuming CWD is project root when running.

    print(f"自动遍历所有模型反馈目录: {os.path.abspath(BASE_DATA_DIRECTORY)}")
    
    # Check if the path exists relative to the current working directory
    if not os.path.isdir(BASE_DATA_DIRECTORY):
        # Attempt to construct path from a hypothetical project root if script is nested
        # This is a common pattern; adjust if your project root detection is different
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_guess = os.path.abspath(os.path.join(script_dir, "..", "..", "..")) # Assuming src is 3 levels down from v1_lang_shining_project
        potential_base_dir = os.path.join(project_root_guess, "v1_lang_shining_project/experiment/MLLMS/feedbacks/")
        
        if os.path.isdir(potential_base_dir):
            BASE_DATA_DIRECTORY = potential_base_dir
            print(f"Adjusted BASE_DATA_DIRECTORY to: {os.path.abspath(BASE_DATA_DIRECTORY)}")
        else:
            print(f"Error: BASE_DATA_DIRECTORY '{os.path.abspath(BASE_DATA_DIRECTORY)}' (and guessed path '{os.path.abspath(potential_base_dir)}') does not exist or is not a directory.")
            print(f"Current working directory: {os.getcwd()}")
            exit()


    print(f"当前工作目录: {os.getcwd()}")
    print(f"BASE_DATA_DIRECTORY绝对路径: {os.path.abspath(BASE_DATA_DIRECTORY)}")


    if not os.path.isdir(BASE_DATA_DIRECTORY):
        print(f"Error: BASE_DATA_DIRECTORY '{os.path.abspath(BASE_DATA_DIRECTORY)}' does not exist or is not a directory.")
        exit()

    # 初始化NLP模型（全局只加载一次）
    print("加载zero-shot分类模型...")
    try:
        analyzer = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"模型加载成功，运行于 {'GPU' if torch.cuda.is_available() else 'CPU'}。");
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保首次下载时有网络连接，并已正确安装PyTorch。")
        exit()

    # 标签定义与映射（与原脚本一致）
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

    # 遍历所有模型反馈子目录
    for model_name in sorted(os.listdir(BASE_DATA_DIRECTORY)):
        model_dir = os.path.join(BASE_DATA_DIRECTORY, model_name)
        if not os.path.isdir(model_dir):
            continue  # 跳过非目录项
        print(f"\n==== 开始分析模型: {model_name} ====")
        # MODEL_FEEDBACK_DIR = model_dir # This was the old concept
        # Output paths are now relative to the model_dir itself (which is already absolute or correct relative to CWD)
        OUTPUT_RESULTS_BASE_DIR = os.path.join(model_dir, "analysis_results")
        OUTPUT_CSV_FILE_MASTER = os.path.join(OUTPUT_RESULTS_BASE_DIR, f"{model_name}_features_consolidated.csv")
        PER_SUBDIR_OUTPUT_DIR = os.path.join(OUTPUT_RESULTS_BASE_DIR, "per_subdir_feedback_features")

        # 创建输出目录
        for d in [OUTPUT_RESULTS_BASE_DIR, PER_SUBDIR_OUTPUT_DIR]:
            if d and not os.path.exists(d):
                os.makedirs(d)
                print(f"创建输出目录: {d}")

        # 递归查找所有.txt文件
        feedback_files_found = []
        # Walk model_dir directly, not MODEL_FEEDBACK_DIR which was an old variable
        for root_dir, _, files in os.walk(model_dir):
            # print(f"遍历目录: {os.path.abspath(root_dir)}") # Can be verbose
            # 跳过analysis_results等输出目录
            if OUTPUT_RESULTS_BASE_DIR in os.path.abspath(root_dir):
                continue
            for file in files:
                if file.endswith(".txt"):
                    feedback_files_found.append(os.path.join(root_dir, file))
        if not feedback_files_found:
            print(f"未找到.txt文件: {model_dir}")
            continue
        print(f"共发现{len(feedback_files_found)}个反馈文件于 {model_name}，开始处理...")

        headers_written_files = set()
        for file_path in tqdm(feedback_files_found, desc=f"[{model_name}] 处理反馈文本"):
            # relative_file_path should be relative to model_dir for consistency in output 'file_id'
            relative_file_path = os.path.relpath(file_path, model_dir)
            critique_text = load_critique_from_txt(file_path)
            if critique_text:
                extracted_stance_data = {}
                extracted_features_data = {}
                extracted_quality_data = {}
                try:
                    stance_results = analyzer(critique_text, candidate_labels=stance_labels_en, multi_label=False)
                    en_stance_label = stance_results['labels'][0]
                    extracted_stance_data = {
                        "label_en": en_stance_label,
                        "label_zh": stance_label_map.get(en_stance_label, ""),
                        "score": stance_results['scores'][0]
                    }
                    feature_results = analyzer(critique_text, candidate_labels=feature_labels_en, multi_label=True)
                    for label, score in zip(feature_results['labels'], feature_results['scores']):
                        extracted_features_data[label] = {
                            "score": score,
                            "zh": feature_label_map.get(label, "")
                        }
                    quality_results = analyzer(critique_text, candidate_labels=quality_labels_en, multi_label=True)
                    for label, score in zip(quality_results['labels'], quality_results['scores']):
                        extracted_quality_data[label] = {
                            "score": score,
                            "zh": quality_label_map.get(label, "")
                        }
                    current_feedback_data = {
                        "file_id": relative_file_path,
                        "text_preview": critique_text[:200] + "..." if len(critique_text) > 200 else critique_text,
                        "stance_label_en": extracted_stance_data.get('label_en'),
                        "stance_label_zh": extracted_stance_data.get('label_zh'),
                        "stance_score": extracted_stance_data.get('score'),
                        "features": str(extracted_features_data),
                        "quality": str(extracted_quality_data)
                    }
                    df_single_row = pd.DataFrame([current_feedback_data])
                    # 合并主CSV
                    master_file_exists = os.path.isfile(OUTPUT_CSV_FILE_MASTER)
                    write_header_master = not master_file_exists or OUTPUT_CSV_FILE_MASTER not in headers_written_files
                    df_single_row.to_csv(OUTPUT_CSV_FILE_MASTER, mode='a', header=write_header_master, index=False, encoding='utf-8-sig')
                    if write_header_master:
                        headers_written_files.add(OUTPUT_CSV_FILE_MASTER)
                    # 分子目录CSV
                    path_parts = os.path.normpath(relative_file_path).split(os.sep)
                    subdir_identifier = path_parts[0] if len(path_parts) > 1 else "root_files"
                    safe_subdir_identifier = "".join(c for c in subdir_identifier if c.isalnum() or c in (' ', '_', '-')).rstrip()
                    if not safe_subdir_identifier:
                        safe_subdir_identifier = "misc_subdir"
                    subdir_csv_path = os.path.join(PER_SUBDIR_OUTPUT_DIR, safe_subdir_identifier + ".csv")
                    subdir_file_exists = os.path.isfile(subdir_csv_path)
                    write_header_subdir = not subdir_file_exists or subdir_csv_path not in headers_written_files
                    df_single_row.to_csv(subdir_csv_path, mode='a', header=write_header_subdir, index=False, encoding='utf-8-sig')
                    if write_header_subdir:
                        headers_written_files.add(subdir_csv_path)
                except Exception as e:
                    tqdm.write(f"NLP分析出错: {relative_file_path}: {e}")
            # else: 错误已在load_critique_from_txt中输出
        print(f"模型[{model_name}]处理完成。主结果: {os.path.abspath(OUTPUT_CSV_FILE_MASTER)}\n分子目录结果: {os.path.abspath(PER_SUBDIR_OUTPUT_DIR)}")
    print("\n全部模型分析完毕。脚本结束。") 