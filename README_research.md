# VULCA Framework - Research Branch

This is the **research branch** containing the complete experimental framework, all ablation studies, and full dataset for the VULCA (Vision-Understanding and Language-based Cultural Adaptability) framework.

## 📁 Branch Contents

### experiments/
Complete experimental code including:
- **preprocessing/**: 14 versions of image preprocessing algorithms
  - v1.0_basic: Initial sliding window implementation
  - v1.2_saliency: Added saliency detection
  - v1.5_adaptive: Adaptive window with dynamic stride
- **drive-download-20250514T190952Z-1-002/**: Original MLLM evaluation code
- **v1_lang_shining_project/**: Full benchmark pipeline

### full_data/
Complete dataset (~50MB):
- **mllm_outputs/**: All MLLM generated critiques (1.2MB)
  - gemini2.5: Complete 12 months evaluations
  - qwen2.5: Full Qwen model outputs
  - llama3.1: LLaMA model results
- **human_expert/**: Human expert annotations and analysis
- **paintings/**: High-resolution Qing Dynasty paintings
- **processed_slices/**: Pre-processed image slices

### notebooks/
Jupyter notebooks for analysis and visualization:
- Data exploration
- Preprocessing evolution analysis
- Evaluation metrics computation
- Paper figure generation

## 🚀 Running Full Experiments

```bash
# Install full dependencies
pip install -r requirements-full.txt

# Run complete benchmark pipeline
cd experiments/v1_lang_shining_project/experiment/Final_research
python run_full_benchmark_pipeline.py

# Run specific preprocessing version
cd experiments/preprocessing/v1.5_adaptive
python noAI1.5.py --input_dir /path/to/images
```

## 📊 Ablation Studies

### Persona Ablation
```bash
cd experiments/drive-download-20250514T190952Z-1-002/MLLMS
python src/generate_mllm_critiques.py --ablation persona
```

### Window Size Ablation
```bash
cd experiments/preprocessing/comparison
python resultcompare.py --compare-window-sizes
```

## 🔬 Reproducing Paper Results

To reproduce the exact results from our EMNLP 2025 paper:

1. **Feature Extraction**:
   ```bash
   cd experiments/v1_lang_shining_project/experiment/Final_research
   python run_full_benchmark_pipeline.py --phase extraction
   ```

2. **Analysis**:
   ```bash
   python run_full_benchmark_pipeline.py --phase analysis
   ```

3. **Visualization**:
   ```bash
   python run_full_benchmark_pipeline.py --phase visualization
   ```

## 📈 Performance Metrics

Our complete evaluation shows:
- Semantic similarity: 0.78 average cosine similarity with human experts
- EMD distance: 0.23 average Earth Mover's Distance
- Profile alignment: 82% accuracy in persona classification

## 🔗 Links

- **Main Branch**: [Streamlined implementation](https://github.com/yha9806/EMNLP2025-VULCA/tree/main)
- **Paper**: [EMNLP 2025 Proceedings](link-to-paper)
- **Dataset DOI**: 10.5281/zenodo.XXXXXX

## 📝 Citation

If you use the complete experimental framework, please cite:
```bibtex
@inproceedings{vulca2025,
  title={VULCA: Vision-Understanding and Language-based Cultural Adaptability Framework},
  author={Your Name},
  booktitle={Proceedings of EMNLP 2025},
  year={2025}
}
```

## 🤝 Contact

For questions about the experimental setup or data, please contact: [your-email]