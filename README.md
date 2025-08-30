# VULCA: Vision-Understanding and Language-based Cultural Adaptability Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-red.svg)](https://2025.emnlp.org/)

## 🎯 Branch Navigation

This repository uses a dual-branch structure for clarity:

| Branch | Purpose | Size | Use Case |
|--------|---------|------|-----------|
| **master** (current) | Core implementation | <10MB | Quick reproduction |
| **[research](https://github.com/yha9806/EMNLP2025-VULCA/tree/research)** | Complete experiments | ~50MB | Full validation |

### For Reviewers
- **Quick Start**: Stay on `master` branch, run `./run_experiments.sh`
- **Deep Dive**: `git checkout research` for all experimental code and data
- **Paper Figures**: See `research` branch `/paper_artifacts/` directory

## Abstract

VULCA presents a comprehensive evaluation framework for assessing Multimodal Large Language Models' (MLLMs) cultural understanding capabilities through Chinese painting critique. Using the Qing Dynasty "Twelve Months" painting series as the evaluation domain, we implement persona-guided prompting with 8 distinct cultural perspectives to measure deep cultural comprehension beyond surface-level recognition.

## Quick Start

```bash
# Clone repository
git clone https://github.com/yha9806/EMNLP2025-VULCA.git
cd EMNLP2025-VULCA

# Install dependencies
pip install -r requirements.txt

# Option 1: Quick evaluation (master branch)
python src/vulca.py --image data/samples/january_sample.jpg

# Option 2: Full experiments (research branch)
git checkout research
python experiments/v1_lang_shining_project/experiment/Final_research/run_full_benchmark_pipeline.py
```

## Installation

### Requirements
- Python 3.8-3.10
- CUDA 11.8+
- 16GB+ GPU memory

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline
```bash
bash run_experiments.sh
```

### Run Individual Components
```bash
# Preprocessing
python src/pipeline.py --skip_evaluation --skip_analysis

# Evaluation
python src/evaluate.py --image data/samples/april_sample.jpg

# Analysis
python src/pipeline.py --skip_preprocessing --skip_evaluation
```

## Project Structure

```
├── src/                 # Core implementation
│   ├── evaluate.py      # MLLM evaluation
│   ├── pipeline.py      # Orchestration
│   ├── preprocessing.py # Image processing
│   ├── analysis.py      # Semantic analysis
│   └── utils.py         # Utilities
├── data/               # Data files
│   ├── personas/       # 8 cultural personas
│   ├── knowledge/      # Knowledge base
│   └── samples/        # Example images
├── configs/            # Configuration
│   ├── model_config.yaml
│   └── experiment_config.yaml
├── outputs/            # Results
└── docs/               # Documentation
```

## Key Features

- **Persona-Guided Prompting**: 8 carefully designed cultural perspectives
- **Adaptive Sliding Window**: Multi-scale image processing (2560/1280/640px)
- **Comprehensive Metrics**: Semantic similarity, EMD, profile scores
- **Multimodal Benchmark**: Systematic evaluation across state-of-the-art MLLMs

## Models Evaluated

- Qwen/Qwen2.5-VL-7B-Instruct
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- google/gemini-2.5-pro

## Results

| Model | Semantic Similarity | Profile Alignment | Cultural Score |
|-------|-------------------|------------------|----------------|
| Qwen2.5-VL | 0.82 | 0.76 | 0.79 |
| Llama-Scout | 0.78 | 0.71 | 0.74 |
| Gemini-2.5 | 0.75 | 0.73 | 0.74 |

## Reproducibility

See [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md) for detailed reproduction instructions.

## Citation

```bibtex
@inproceedings{vulca2025emnlp,
  title={VULCA: A Comprehensive Evaluation Framework for Assessing Multimodal Large Language Models' Cultural Understanding through Chinese Art Critique},
  author={[Authors]},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  address={Suzhou, China}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the EMNLP 2025 reviewers for their valuable feedback.

## Contact

- GitHub Issues: [https://github.com/yha9806/EMNLP2025-VULCA/issues](https://github.com/yha9806/EMNLP2025-VULCA/issues)
- Email: [contact information]