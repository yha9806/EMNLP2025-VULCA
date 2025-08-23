# VULCA: Vision-Understanding and Language-based Cultural Adaptability Framework

This repository contains the code and data for our EMNLP 2025 submission: **"VULCA: A Comprehensive Evaluation Framework for Assessing Multimodal Large Language Models' Cultural Understanding through Chinese Art Critique"**.

## 📚 Abstract

VULCA Framework presents a novel approach to evaluating Multimodal Large Language Models (MLLMs) using Chinese painting critique as the evaluation domain. By leveraging the Qing Dynasty "Twelve Months" painting series and implementing persona-guided prompting with 8 distinct cultural perspectives, we assess MLLMs' deep cultural understanding capabilities beyond surface-level recognition.

## 🌟 Key Features

- **Persona-Guided Prompting**: 8 carefully designed cultural personas representing diverse perspectives in Chinese art criticism
- **Adaptive Sliding Window**: Advanced image preprocessing for handling high-resolution traditional Chinese paintings
- **Comprehensive Evaluation Pipeline**: End-to-end system from image processing to semantic alignment analysis
- **Multimodal Benchmark**: Systematic evaluation across multiple state-of-the-art MLLMs

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yha9806/EMNLP2025-VULCA.git
cd EMNLP2025-VULCA

# Install dependencies
pip install torch>=2.0.0
pip install vllm
pip install transformers>=4.35.0
pip install sentence-transformers
pip install opencv-python
pip install scikit-image
pip install pandas numpy matplotlib seaborn
```

## 🚀 Quick Start

### 1. Run the Full Benchmark Pipeline

```bash
cd deployed_content/Experiment/v1_lang_shining_project
python experiment/Final_research/run_full_benchmark_pipeline.py
```

### 2. Generate MLLM Critiques

First, start the vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 16384
```

Then generate critiques:
```bash
cd deployed_content/Experiment/drive-download-20250514T190952Z-1-002/MLLMS
python src/generate_mllm_critiques.py \
  --image_path <path_to_image> \
  --model_name <model_id> \
  --persona_file persona/<persona_name>.md
```

### 3. Image Preprocessing

```bash
cd deployed_content/Experiment/v1_lang_shining_project/experiment/autowindow
python noAI1.5.py
```

## 📂 Project Structure

```
EMNLP2025-VULCA/
├── deployed_content/
│   ├── Experiment/
│   │   ├── drive-download-20250514T190952Z-1-002/
│   │   │   └── MLLMS/             # Core MLLM evaluation system
│   │   │       ├── src/            # Main scripts
│   │   │       ├── prompt/         # Prompt templates
│   │   │       ├── persona/        # 8 cultural personas
│   │   │       └── knowledge_dataset/  # Knowledge base
│   │   └── v1_lang_shining_project/
│   │       └── experiment/
│   │           ├── autowindow/     # Image preprocessing
│   │           └── Final_research/ # Pipeline orchestration
│   └── MHEB-Dataset/               # Multimodal Human Expert Benchmark
├── CLAUDE.md                        # Project documentation
└── README.md                        # This file
```

## 🎯 Evaluation Components

### Cultural Personas
- **Traditional Scholar** (传统文人)
- **Modern Curator** (现代策展人)
- **Art Historian** (艺术史学家)
- **Cultural Preservationist** (文化保护者)
- **Contemporary Artist** (当代艺术家)
- **Art Educator** (艺术教育者)
- **Collector** (收藏家)
- **General Audience** (普通观众)

### Metrics
- **Semantic Similarity**: Cosine similarity with expert embeddings
- **Earth Mover's Distance**: Distribution comparison
- **Profile Scores**: Weighted capability assessment
- **Persona Alignment**: Rule-based matching to expert profiles

## 📊 Results

Our experiments demonstrate that:
- Current MLLMs show varying capabilities in understanding cultural nuances
- Persona-guided prompting significantly enhances critique quality
- The framework effectively distinguishes between surface-level and deep cultural understanding

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{vulca2025emnlp,
  title={VULCA: A Comprehensive Evaluation Framework for Assessing Multimodal Large Language Models' Cultural Understanding through Chinese Art Critique},
  author={[Author Names]},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  address={Suzhou, China}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration, please contact: [email]

## 🙏 Acknowledgments

We thank the reviewers and the EMNLP 2025 organizing committee for their valuable feedback.

---

**Note**: This repository contains the code implementation for our EMNLP 2025 submission. The paper PDF and supplementary materials will be made available upon acceptance.