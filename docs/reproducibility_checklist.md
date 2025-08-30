# VULCA Framework - Reproducibility Checklist

## Environment Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (recommended: RTX 3090, A100)
- **RAM**: 32GB minimum
- **Storage**: 50GB free space

### Software
- **OS**: Ubuntu 20.04+ or Windows 10/11 with WSL2
- **Python**: 3.8-3.10
- **CUDA**: 11.8+
- **Git**: 2.25+

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yha9806/EMNLP2025-VULCA.git
cd EMNLP2025-VULCA
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
```bash
bash data/download_data.sh
```

## Running Experiments

### Quick Start
```bash
make run
```

### Step-by-Step Execution

#### 1. Start vLLM Server
```bash
make server
# Wait for server to start (check http://localhost:8000/health)
```

#### 2. Run Preprocessing
```bash
python src/pipeline.py --skip_evaluation --skip_analysis --skip_visualization
```

#### 3. Run Evaluation
```bash
python src/evaluate.py \
  --image_path data/sample_data/images/sample.png \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct
```

#### 4. Run Full Pipeline
```bash
python src/pipeline.py --model_name Qwen/Qwen2.5-VL-7B-Instruct
```

## Expected Outputs

### Directory Structure After Running
```
outputs/
├── processed_images/     # Preprocessed image slices
├── critiques/           # Generated MLLM critiques
├── analysis/            # Analysis results
├── metrics/             # Evaluation metrics
└── visualizations/      # Generated plots
```

### Key Files
- `outputs/metrics/evaluation_results.json`: Main results
- `outputs/analysis/semantic_similarity.csv`: Similarity scores
- `outputs/visualizations/tsne_plot.png`: t-SNE visualization

## Validation

### Test Installation
```bash
make test
```

### Verify Results
```python
import pandas as pd
results = pd.read_csv('outputs/analysis/results.csv')
print(results.describe())
```

## Common Issues

### Issue 1: CUDA Out of Memory
**Solution**: Reduce batch size in `configs/experiment_config.yaml`

### Issue 2: vLLM Server Connection Error
**Solution**: Ensure server is running on port 8000
```bash
curl http://localhost:8000/health
```

### Issue 3: Missing Dependencies
**Solution**: Update pip and reinstall
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Configuration Options

### Model Selection
Edit `configs/model_config.yaml` to change models

### Experiment Parameters
Edit `configs/experiment_config.yaml` to adjust:
- Batch size
- Number of personas
- Window sizes
- Output directories

## Reproduction Guarantee

Following these steps exactly should produce results within ±2% of reported metrics due to:
- Fixed random seeds (seed=42)
- Deterministic preprocessing
- Version-locked dependencies

## Contact

For issues or questions:
- GitHub Issues: https://github.com/yha9806/EMNLP2025-VULCA/issues
- Email: [contact email]

## Citation

If reproducing this work, please cite:
```bibtex
@inproceedings{vulca2025emnlp,
  title={VULCA: A Comprehensive Evaluation Framework},
  author={[Authors]},
  booktitle={EMNLP 2025},
  year={2025}
}
```