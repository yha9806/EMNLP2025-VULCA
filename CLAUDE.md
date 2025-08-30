# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VULCA Framework (Vision-Understanding and Language-based Cultural Adaptability) - A comprehensive evaluation system for Multimodal Large Language Models (MLLMs) using Chinese painting critique as the evaluation domain. The project implements persona-guided prompting with 8 cultural perspectives to assess MLLMs' cultural understanding capabilities across the Qing Dynasty "Twelve Months" painting series.

## Key Commands

### Running the Full Benchmark Pipeline
```bash
cd deployed_content/Experiment/v1_lang_shining_project
python experiment/Final_research/run_full_benchmark_pipeline.py
# Orchestrates: feature extraction → analysis → visualization
```

### MLLM Critique Generation
```bash
# Start vLLM server first (required for API calls)
python -m vllm.entrypoints.openai.api_server --model <model_name> --port 8000 --trust-remote-code --max-model-len 16384

# Generate critiques with persona enhancement
cd deployed_content/Experiment/drive-download-20250514T190952Z-1-002/MLLMS
python src/generate_mllm_critiques.py \
  --image_path <path_to_image> \
  --output_path feedbacks/<model>/<month>/<output_file> \
  --model_name <model_id> \
  --prompt_file prompt/prompt.md \
  --persona_file persona/<persona_name>.md \
  --knowledge_base knowledge_dataset/knowledge_base.json \
  --model_params '{"max_new_tokens": 2048, "temperature": 0.7}'
```

### Image Preprocessing with Adaptive Sliding Window
```bash
cd deployed_content/Experiment/v1_lang_shining_project/experiment/autowindow
python noAI1.5.py
# Input: D:\Qiufeng\aiart\langshining
# Output: D:\Qiufeng\aiart\Auto_Cropping_langshining
# Creates: 640x640 normalized slices from adaptive windows (2560/1280/640px)
```

### Using the Python Package
```python
from vulca import MLLMEvaluator

evaluator = MLLMEvaluator(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    api_endpoint="http://localhost:8000",
    api_type="vllm"
)

result = evaluator.generate_critique(
    image_path="path/to/image.jpg",
    prompt="Critique this painting",
    persona="traditional_scholar"
)
```

### Paper Compilation (LaTeX)
```bash
cd deployed_content/Paper_writing

# Option 1: Direct compilation
xelatex final.tex     # First pass
bibtex final          # Process bibliography
xelatex final.tex     # Second pass for citations
xelatex final.tex     # Third pass for final formatting

# Option 2: Setup environment first (Windows)
setup_latex_env.bat   # Configure Perl and MiKTeX paths
```

### Running Tests
```bash
# Unit tests for vulca package
python -m pytest vulca/tests/

# Playwright tests for web automation
npm install playwright
node test_playwright.js
```

## Architecture Overview

### Core Components

**1. MLLM Evaluation System** (`deployed_content/Experiment/drive-download-20250514T190952Z-1-002/MLLMS/`)
- `src/generate_mllm_critiques.py`: OpenAI-compatible API integration with vLLM server
- API endpoint: `http://localhost:8000/v1/chat/completions`
- Base64 image encoding with multimodal message formatting
- Dynamic prompt construction: base_prompt + persona + knowledge context

**2. Image Processing Pipeline** (`deployed_content/Experiment/v1_lang_shining_project/experiment/autowindow/`)
- `noAI1.5.py`: Adaptive sliding window implementation
- Saliency detection: Sobel edge density + LBP texture complexity
- Dynamic stride adjustment: 0.9x (low), 0.75x (medium), 0.5x (high) window size
- Output normalization: All slices resized to 640x640px

**3. Analysis Pipeline** (`deployed_content/Experiment/v1_lang_shining_project/experiment/Final_research/`)
- `run_full_benchmark_pipeline.py`: Orchestrates entire evaluation workflow
- Feature extraction using BAAI/bge-large-zh-v1.5 (1024-dim embeddings)
- Statistical analysis: EMD, cosine similarity, profile matching
- Visualization: t-SNE, UMAP, distribution plots

**4. Python Package** (`vulca/`)
- `core/mllm_evaluator.py`: Main evaluator class for API interactions
- `core/persona_manager.py`: Manages 8 cultural personas
- `preprocessing/adaptive_window.py`: Image slicing logic
- `analysis/semantic_analyzer.py`: Embedding and similarity computation
- `knowledge/knowledge_base.py`: Hierarchical knowledge retrieval

### Data Flow
```
Raw Paintings → Adaptive Slicing → Base64 Encoding → vLLM API
                                                         ↓
Knowledge Base + Persona → Enhanced Prompt → MLLM Generation
                                                         ↓
Generated Critiques → Feature Extraction → Profile Scoring
                                                         ↓
Semantic Vectors → Alignment Analysis → Visualization & Stats
```

### Directory Structure
```
deployed_content/
├── Paper_writing/              # LaTeX paper source and figures
│   ├── final.tex               # Main paper document (requires XeLaTeX)
│   ├── appendix.tex            # Supplementary material
│   ├── references.bib          # Bibliography
│   └── picture/                # Figures and visualizations
├── Experiment/
│   ├── drive-download-20250514T190952Z-1-002/
│   │   ├── MLLMS/              # Core MLLM evaluation
│   │   │   ├── src/            # Main scripts
│   │   │   ├── prompt/         # Prompt templates
│   │   │   ├── persona/        # 8 persona definitions
│   │   │   ├── knowledge_dataset/  # Knowledge base
│   │   │   └── feedbacks/      # Generated critiques by model
│   │   └── proceed/            # Processed image slices
│   └── v1_lang_shining_project/
│       └── experiment/
│           ├── autowindow/     # Image preprocessing
│           ├── human_expert/   # Benchmark analysis
│           └── Final_research/ # Pipeline orchestration
vulca/                          # Python package for VULCA framework
├── core/                       # Core evaluation modules
├── preprocessing/              # Image processing
├── analysis/                   # Statistical analysis
└── knowledge/                  # Knowledge management
```

## Critical Implementation Details

### vLLM Server Configuration
```bash
# Start server with model
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9

# Windows PATH issue (from .cursor/rules/vLLM.mdc):
# If vllm command not found, add to PATH:
# C:\Users\<username>\AppData\Roaming\Python\Python310\Scripts
```

### Persona Integration Logic
1. Load persona markdown from `persona/{name}.md`
2. Query knowledge base for relevant context (4 categories: chinese_landscape_painting, qing_court_painting, giuseppe_castiglione, twelve_months_paintings)
3. Construct prompt: `base_prompt + persona_description + knowledge_snippets`
4. Send to vLLM with image as base64 in OpenAI format

### Image Processing Parameters
- **Saliency Thresholds**: Low (0.25), High (0.6)
- **Stride Factors**: 0.9x (low), 0.75x (medium), 0.5x (high)
- **Window Sizes**: 2560px, 1280px, 640px (adaptive based on image size)
- **Output**: All slices normalized to 640x640px

### Models Tested
- **Qwen/Qwen2.5-VL-7B-Instruct**: Best Chinese understanding
- **meta-llama/Llama-4-Scout-17B-16E-Instruct**: Strong reasoning
- **meta-llama/Llama-3.1-8B-Instruct**: Baseline comparison
- **google/gemini-2.5-pro**: Commercial API baseline

### LaTeX Compilation Requirements
```bash
# Required for Chinese characters
xelatex final.tex     # First pass
bibtex final          # Bibliography
xelatex final.tex     # Citations
xelatex final.tex     # Final

# Windows-specific paths (from setup_latex_env.bat):
# Perl: C:\Users\<username>\Downloads\strawberry-perl-5.40.2.1-64bit-portable\perl\bin
# MiKTeX: C:\Users\<username>\AppData\Local\Programs\MiKTeX\miktex\bin\x64
```
- Font: Microsoft YaHei (must be installed)
- Packages: fontspec, xeCJK, microtype
- Compiler: XeLaTeX (not pdfLaTeX)

## Key Python Dependencies
```
vllm                    # Model serving
transformers>=4.35.0    # Model loading
torch>=2.0.0           # Deep learning
sentence-transformers   # BAAI/bge-large-zh-v1.5 embeddings
opencv-python          # Image processing
scikit-image           # LBP texture analysis
pandas, numpy          # Data manipulation
matplotlib, seaborn    # Visualization
tqdm                   # Progress bars
requests               # API calls
```

## Environment Setup
```bash
# GPU configuration (Windows)
set CUDA_VISIBLE_DEVICES=0

# Model cache paths
set HF_HOME=D:\models\huggingface
set TRANSFORMERS_CACHE=D:\models\transformers

# vLLM configuration
set VLLM_PORT=8000
set VLLM_HOST=0.0.0.0

# API tokens (use environment variables)
set HF_TOKEN=your_huggingface_token_here
```

## Evaluation Metrics
- **Semantic Similarity**: Cosine similarity with human expert embeddings (BAAI/bge-large-zh-v1.5)
- **Earth Mover's Distance (EMD)**: Distribution comparison across feature dimensions
- **Profile Scores**: Weighted capability assessment (stance: 0.3, focus: 0.4, quality: 0.3)
- **Persona Alignment**: Rule-based matching to 5 expert profiles using profile_score thresholds

## Analysis Pipeline Workflow

The full benchmark pipeline (`run_full_benchmark_pipeline.py`) executes these phases sequentially:

1. **Feature Extraction**: Process MLLM feedbacks into 1024-dim semantic vectors
2. **Data Consolidation**: Merge features from all models and human experts
3. **Statistical Analysis**: Compute similarity metrics and distribution comparisons
4. **Visualization Generation**: Create t-SNE, UMAP, and distribution plots
5. **Report Generation**: Output analysis results to CSV and JSON formats

Expected output structure:
```
result/
├── analysis_results/           # Statistical analysis outputs
├── eda_plots/                  # Visualization figures
├── human_expert_features_consolidated.csv  # Benchmark data
└── all_models_features_consolidated.csv    # Combined MLLM features
```

## Persona System Clarification

### 8 Cultural Persona Cards (Experimental Interventions)
These are used as prompts to guide MLLM generation during experiments:

1. **郭熙 (Guo Xi)** - Song Dynasty landscape painter perspective
2. **苏轼 (Su Shi)** - Song Dynasty literati and poet perspective
3. **托马斯修士 (Brother Thomas)** - Medieval European religious art perspective
4. **约翰·罗斯金 (John Ruskin)** - Victorian art criticism perspective
5. **佐拉妈妈 (Mama Zola)** - African cultural and community perspective
6. **阿里斯·索恩博士 (Dr. Aris Thorne)** - Contemporary Western academic perspective
7. **埃琳娜·佩特洛娃教授 (Professor Elena Petrova)** - Russian art history perspective
8. **冈仓天心 (Okakura Kakuzo)** - Japanese aesthetic philosophy perspective

### 5 Core Personas (Evaluation Categories)
These are data-driven categories for classifying generated critiques:

1. **Comprehensive Analyst (博学通论型)** - Broad analytical coverage
2. **Historically Focused Critic (历史考据型)** - Historical context emphasis
3. **Technique & Style Focused Critic (技艺风格型)** - Technical analysis focus
4. **Theory & Comparison Focused Critic (理论比较型)** - Theoretical framework
5. **General Descriptive Profile (泛化描述型)** - General descriptive approach

## Feature Dimensions

The framework uses a **47-dimensional feature vector** comprising:
- **38 Primary Labels**:
  - 10 Evaluative Stance labels
  - 17 Feature Focus labels
  - 11 Commentary Quality labels
- **9 Derived Dimensions**:
  - 5 Profile alignment scores
  - 4 Supplementary analytical dimensions

## MHEB Dataset Details

- **163 expert commentaries** from 9 distinguished art historians
- **Expert sources**: Xue Yongnian (薛永年, 17 texts), Wang Di (汪涤, 28 texts), Yang Danxia (杨丹霞, 28 texts), Nie Chongzheng (聂崇正, 15 texts), Shan Guoqiang (单国强, 18 texts), Li Shi (李湜, 17 texts), Xu Jianrong (徐建融, 17 texts), Zhu Wanzhang (朱万章, 11 texts), Chen Yunru (陈韵如, 12 texts)
- **Annotation**: 3 annotators with graduate-level training in Chinese art history
- **Inter-Annotator Agreement**: Fleiss' kappa = 0.78, ICC = 0.82