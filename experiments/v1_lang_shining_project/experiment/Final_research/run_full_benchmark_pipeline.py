import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import glob
import shutil
import logging
import subprocess
import fnmatch

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_full_benchmark_pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 新增详细日志 handler
full_detail_log_path = "full_benchmark_detail.log"
detail_handler = logging.FileHandler(full_detail_log_path, encoding='utf-8')
detail_handler.setLevel(logging.INFO)
detail_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
detail_handler.setFormatter(detail_formatter)
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(full_detail_log_path) for h in logging.getLogger().handlers):
    logging.getLogger().addHandler(detail_handler)

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLLMS_FEEDBACKS_DIR = PROJECT_ROOT / "experiment/MLLMS/feedbacks"
MLLMS_RESULT_DIR = PROJECT_ROOT / "experiment/MLLMS/result"
HUMAN_EXPERT_CSV = PROJECT_ROOT / "result/human_expert_features_consolidated.csv"
ANALYSIS_RESULTS_DIR = PROJECT_ROOT / "result/analysis_results"
EDA_PLOTS_DIR = PROJECT_ROOT / "result/eda_plots"

# ========== 合并多模型 features_consolidated.csv ==========
def merge_all_model_features(output_path):
    logging.info("开始合并所有模型 features_consolidated.csv ...")
    all_csv_files = glob.glob(str(MLLMS_FEEDBACKS_DIR / "*/analysis_results/*_features_consolidated.csv"))
    dfs = []
    for csv_file in all_csv_files:
        model_name = Path(csv_file).parts[-3]
        try:
            logging.info(f"[详细日志] 读取模型特征文件: {csv_file}")
            df = pd.read_csv(csv_file)
            logging.info(f"[详细日志] 文件 {csv_file} 行数: {len(df)}, 字段: {list(df.columns)}")
            df['model_name'] = model_name
            # persona_id 兼容性处理
            if 'persona_id' not in df.columns:
                # 尝试从 file_id 推断
                if 'file_id' in df.columns:
                    df['persona_id'] = df['file_id'].apply(lambda x: str(x).split('/')[0] if pd.notna(x) and '/' in str(x) else 'unknown')
                else:
                    df['persona_id'] = 'unknown'
            df['source_type'] = 'mllm'
            dfs.append(df)
            logging.info(f"已加载: {csv_file} ({len(df)})")
        except Exception as e:
            logging.warning(f"跳过 {csv_file}: {e}")
    if not dfs:
        logging.error("未找到任何模型 features_consolidated.csv，合并失败！")
        return False
    merged_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"[详细日志] 合并后总行数: {len(merged_df)}, 字段: {list(merged_df.columns)}")
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"合并完成，输出到: {output_path} (共{len(merged_df)}条)")
    logging.info(f"[详细日志] 合并输出文件: {output_path}")
    return True

# ========== 合并 human_expert 数据 ==========
def check_human_expert_csv():
    if not HUMAN_EXPERT_CSV.exists():
        logging.error(f"未找到 human_expert_features_consolidated.csv: {HUMAN_EXPERT_CSV}")
        return False
    try:
        logging.info(f"[详细日志] 读取 human_expert 文件: {HUMAN_EXPERT_CSV}")
        df = pd.read_csv(HUMAN_EXPERT_CSV)
        logging.info(f"[详细日志] 文件 {HUMAN_EXPERT_CSV} 行数: {len(df)}, 字段: {list(df.columns)}")
        if 'persona_id' not in df.columns:
            df['persona_id'] = 'human_expert'
        if 'source_type' not in df.columns:
            df['source_type'] = 'human_expert'
        df.to_csv(HUMAN_EXPERT_CSV, index=False, encoding='utf-8-sig')
        logging.info(f"已检查并标准化 human_expert 数据: {HUMAN_EXPERT_CSV}")
        return True
    except Exception as e:
        logging.error(f"human_expert_features_consolidated.csv 加载失败: {e}")
        return False

# ========== 主流程 ==========
def run_subprocess_py(script_path, desc):
    logging.info(f"[自动化] 开始执行: {desc} -> {script_path}")
    logging.info(f"[详细日志] 调用子脚本: {script_path}, 描述: {desc}")
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=True, text=True)
        logging.info(f"[自动化] {desc} 完成。输出如下：\n{result.stdout}")
        if result.stderr:
            logging.warning(f"[自动化] {desc} 警告/错误输出：\n{result.stderr}")
        logging.info(f"[详细日志] 子脚本 {script_path} 返回码: {result.returncode}")
        logging.info(f"[详细日志] 子脚本 {script_path} STDOUT: {result.stdout}")
        if result.stderr:
            logging.info(f"[详细日志] 子脚本 {script_path} STDERR: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"[自动化] {desc} 执行失败！\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        logging.error(f"[详细日志] 子脚本 {script_path} 执行失败，返回码: {e.returncode}")
        logging.error(f"[详细日志] 子脚本 {script_path} STDOUT: {e.stdout}")
        logging.error(f"[详细日志] 子脚本 {script_path} STDERR: {e.stderr}")
        raise

def copy_outputs_to_paper_dirs():
    # 目标路径
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DOCS_DATA_DIR = PROJECT_ROOT / "docs/paper_writing/data"
    DOCS_PIC_DIR = PROJECT_ROOT / "docs/paper_writing/picture"
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_PIC_DIR.mkdir(parents=True, exist_ok=True)
    # 源路径
    ANALYSIS_RESULTS_DIR = PROJECT_ROOT / "result/analysis_results"
    EDA_PLOTS_DIR = PROJECT_ROOT / "result/eda_plots"
    # 数据文件类型
    data_patterns = ['*.csv', '*.tsv', '*.txt', '*.xlsx', '*.json']
    # 图片文件类型
    pic_patterns = ['*.png', '*.jpg', '*.jpeg', '*.svg']
    # 复制数据文件（递归遍历所有子目录，保持文件名唯一）
    for pattern in data_patterns:
        for src in ANALYSIS_RESULTS_DIR.rglob(pattern):
            dst = DOCS_DATA_DIR / src.name
            shutil.copy2(src, dst)
            size = src.stat().st_size if src.exists() else 0
            logging.info(f"[输出同步] 数据文件已复制: {src} -> {dst}")
            logging.info(f"[详细日志] 复制数据文件: {src} (大小: {size} bytes) -> {dst}")
    # 复制图片文件（递归遍历所有子目录，保持文件名唯一）
    for pattern in pic_patterns:
        for src in EDA_PLOTS_DIR.rglob(pattern):
            dst = DOCS_PIC_DIR / src.name
            shutil.copy2(src, dst)
            size = src.stat().st_size if src.exists() else 0
            logging.info(f"[输出同步] 图片文件已复制: {src} -> {dst}")
            logging.info(f"[详细日志] 复制图片文件: {src} (大小: {size} bytes) -> {dst}")

def main(args):
    logging.info("[详细日志] ====== 全流程主控脚本开始 ======")
    # 1. 合并所有模型 features_consolidated.csv
    merged_csv_path = MLLMS_RESULT_DIR / "mllms_features_consolidated.csv"
    logging.info(f"[详细日志] 阶段1: 合并模型特征，输出路径: {merged_csv_path}")
    merge_ok = merge_all_model_features(merged_csv_path)
    if not merge_ok:
        logging.error("模型特征合并失败，终止流程。")
        logging.info("[详细日志] 阶段1: 合并模型特征 失败，流程终止")
        return
    logging.info("[详细日志] 阶段1: 合并模型特征 完成")
    # 2. 检查 human_expert 数据
    logging.info(f"[详细日志] 阶段2: 检查 human_expert 数据，路径: {HUMAN_EXPERT_CSV}")
    human_ok = check_human_expert_csv()
    if not human_ok:
        logging.error("human_expert 数据检查失败，终止流程。")
        logging.info("[详细日志] 阶段2: 检查 human_expert 失败，流程终止")
        return
    logging.info("[详细日志] 阶段2: 检查 human_expert 完成")
    # 3. 自动化分析与降维
    script_dir = Path(__file__).resolve().parent
    phase3_dir = PROJECT_ROOT / "experiment/human_expert/src/phase3"
    phase4_dir = PROJECT_ROOT / "experiment/human_expert/src/phase4"
    phase5_dir = PROJECT_ROOT / "experiment/human_expert/src/phase5"
    # 3.1 主分析 part1
    logging.info(f"[详细日志] 阶段3.1: 主分析 part1 (统计/降维/合并)")
    run_subprocess_py(phase3_dir / "phase_1_1_benchmark_script_part1.py", "主分析 part1 (统计/降维/合并)")
    logging.info(f"[详细日志] 阶段3.1: 主分析 part1 完成")
    # 3.2 主分析 part2
    logging.info(f"[详细日志] 阶段3.2: 主分析 part2 (profile 匹配/聚类/比例)")
    run_subprocess_py(phase4_dir / "phase_1_1_benchmark_script_part2_candidate_selection.py", "主分析 part2 (profile 匹配/聚类/比例)")
    logging.info(f"[详细日志] 阶段3.2: 主分析 part2 完成")
    # 4. 语义向量与中心
    logging.info(f"[详细日志] 阶段4: 生成作者语义向量与中心")
    run_subprocess_py(phase5_dir / "generate_author_semantic_vectors.py", "生成作者语义向量与中心")
    run_subprocess_py(phase5_dir / "generate_semantic_vectors.py", "生成 profile 语义向量与中心")
    logging.info(f"[详细日志] 阶段4: 语义向量与中心 完成")
    # 5. 可视化
    logging.info(f"[详细日志] 阶段5: 可视化分析")
    run_subprocess_py(phase5_dir / "visualize_author_semantic_space.py", "可视化作者语义空间")
    run_subprocess_py(phase5_dir / "visualize_semantic_space.py", "可视化 profile 语义空间")
    run_subprocess_py(PROJECT_ROOT / "experiment/human_expert/src/visualize_composite_analysis.py", "复合分析可视化")
    logging.info(f"[详细日志] 阶段5: 可视化分析 完成")
    # 6. 复制输出到 paper_writing 目录
    logging.info(f"[详细日志] 阶段6: 复制输出文件到 paper_writing 目录")
    copy_outputs_to_paper_dirs()
    logging.info(f"[详细日志] 阶段6: 输出文件复制 完成")
    logging.info("[详细日志] ====== 全流程主控脚本结束，所有分析与可视化已自动完成 ======")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="一键式多模型+human_expert 全流程分析主控脚本")
    # 可扩展参数，如 --overwrite、--models、--visualize 等
    args = parser.parse_args()
    main(args) 