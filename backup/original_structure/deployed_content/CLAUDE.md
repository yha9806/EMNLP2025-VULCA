# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VULCA Framework (Vision-Understanding and Language-based Cultural Adaptability) - A comprehensive evaluation system for Multimodal Large Language Models (MLLMs) using Chinese painting critique as the evaluation domain. The project implements persona-guided prompting with 8 cultural perspectives to assess MLLMs' cultural understanding capabilities.

## Key Commands

### Running the Full Benchmark Pipeline
```bash
python I:/EMNLP2025/deployed_content/Experiment/v1_lang_shining_project/experiment/Final_research/run_full_benchmark_pipeline.py
```
This orchestrates the entire evaluation pipeline including feature extraction, analysis, and visualization.

### MLLM Critique Generation
```bash
# Start vLLM server first
vllm serve <model_name> --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len 16384

# Generate critiques
python experiment/MLLMS/src/generate_mllm_critiques.py --mode [baseline|persona] --model <model_name> --painting_id <id> --image_path <path>
```

### Image Processing
```bash
python experiment/autowindow/noAI1.5.py
# Processes images with adaptive sliding window (2560/1280/640px windows)
# Input: D:\Qiufeng\aiart\langshining
# Output: D:\Qiufeng\aiart\Auto_Cropping_langshining
```

### LaTeX Compilation
```bash
cd I:/EMNLP2025/deployed_content/Paper_writing
xelatex final.tex
# Or use setup_latex_env.bat to configure environment first
```

## Architecture Overview

### Core Components

**1. MLLM Evaluation System** (`experiment/MLLMS/`)
- `src/generate_mllm_critiques.py`: Main API integration with vLLM server
- OpenAI-compatible API endpoint: `http://localhost:8000/v1/chat/completions`
- Handles Base64 image encoding and structured prompt generation
- Supports both baseline and persona-enhanced modes

**2. Image Processing Pipeline** (`experiment/autowindow/`)
- `noAI1.5.py`: Adaptive sliding window implementation
- Combines Sobel edge detection + LBP texture analysis for saliency mapping
- Dynamic stride adjustment based on information density (0.5x to 0.9x window size)
- Outputs: 640x640 normalized slices from variable source windows

**3. Analysis Pipeline** (`experiment/human_expert/src/`)
- Phase 1-5 modular architecture for progressive analysis
- Feature extraction → Statistical analysis → Semantic vectorization → Visualization
- BAAI/bge-large-zh-v1.5 model for 1024-dim semantic embeddings
- Composite evaluation across 3 dimensions: Stance × Focus × Quality

**4. Knowledge Integration**
- `knowledge_dataset/knowledge_base.json`: Hierarchical domain knowledge
- Categories: chinese_landscape_painting, qing_court_painting, giuseppe_castiglione, twelve_months_paintings
- Dynamic retrieval based on persona and painting context

### Directory Structure
```
I:/EMNLP2025/deployed_content/
├── Experiment/
│   ├── v1_lang_shining_project/
│   │   ├── experiment/
│   │   │   ├── MLLMS/           # MLLM evaluation
│   │   │   ├── human_expert/    # Benchmark analysis
│   │   │   ├── autowindow/      # Image processing
│   │   │   └── Final_research/  # Pipeline orchestration
│   │   └── docs/paper_writing/  # Documentation
│   └── drive-download-20250514T190952Z-1-002/
│       └── MLLMS/               # Original experiment data
└── Paper_writing/               # LaTeX source files
```

## Critical Implementation Details

### vLLM Server Configuration
- Max model length: 16384 tokens
- Trust remote code enabled for custom models
- Supports concurrent requests with batching
- Models tested: Qwen2.5-VL-7B, Qwen2.5-Omni-7B, Janus-Pro-7B, deepseek-vl2

### Persona System (8 Cultural Perspectives)
1. 郭熙 (Guo Xi) - Song Dynasty painter
2. 苏轼 (Su Shi) - Song Dynasty literatus
3. 托马斯修士 (Brother Thomas) - Medieval European monk
4. 约翰·罗斯金 (John Ruskin) - Victorian art critic
5. 佐拉妈妈 (Mama Zola) - African cultural perspective
6. 阿里斯·索恩博士 (Dr. Aris Thorne) - Contemporary Western academic
7. 埃琳娜·佩特洛娃教授 (Professor Elena Petrova) - Russian art historian
8. 冈仓天心 (Okakura Kakuzo) - Japanese aesthetician

### Evaluation Metrics
- **Semantic Similarity**: Cosine similarity in 1024-dim space
- **Earth Mover's Distance (EMD)**: Distribution comparison
- **Profile Matching**: 5-dimensional capability scores
- **Clustering Analysis**: t-SNE/UMAP for visualization

### LaTeX Requirements
- Compiler: XeLaTeX (required for Chinese fonts)
- Main document: final.tex
- Fonts: Microsoft YaHei for CJK text
- Style files: acl.sty, emnlp2024.sty
- Bibliography: acl_natbib.bst with references.bib

## Data Flow

1. **Image Preparation**: Raw paintings → Adaptive sliding → 640x640 slices
2. **Critique Generation**: Slices + Persona + Prompt → MLLM API → Text critiques
3. **Feature Extraction**: Critiques → Semantic vectors (1024-dim)
4. **Analysis**: Vectors → Statistical comparison → Visualization
5. **Output**: Analysis results → LaTeX tables/figures → PDF paper

## Key Python Dependencies
- vllm (for model serving)
- transformers, sentence-transformers (for embeddings)
- opencv-python, scikit-image (for image processing)
- pandas, numpy (for data analysis)
- matplotlib, seaborn (for visualization)
- torch (PyTorch backend)

## Important File Paths
- Human expert benchmark: `result/human_expert_features_consolidated.csv`
- MLLM outputs: `experiment/MLLMS/feedbacks/<model_name>/`
- Analysis results: `result/analysis_results/`
- Visualization outputs: `result/eda_plots/`
- Paper figures: `Paper_writing/picture/`

## Notes
- The project uses Windows paths (I:/ drive)
- vLLM server must be running before critique generation
- Image processing is memory-intensive for large paintings
- XeLaTeX compilation requires proper font installation