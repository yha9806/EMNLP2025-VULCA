#!/usr/bin/env python
"""
VULCA Framework - Full Benchmark Pipeline
Orchestrates the complete evaluation workflow
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def setup_paths(config_path: str = "configs/hyperparams.yaml") -> Dict[str, Path]:
    """Setup project paths."""
    project_root = Path(__file__).parent.parent
    return {
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'outputs_dir': project_root / 'outputs',
        'configs_dir': project_root / 'configs',
        'src_dir': project_root / 'src'
    }


def run_preprocessing(image_dir: str, output_dir: str) -> None:
    """Run image preprocessing with adaptive sliding window."""
    logging.info("Starting image preprocessing...")
    from preprocess import process_image
    import os
    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(image_dir, image_file)
            process_image(image_path, output_dir)
    logging.info("Image preprocessing complete")


def run_evaluation(images_dir: str, model_name: str, personas_dir: str) -> None:
    """Run MLLM evaluation on processed images."""
    logging.info(f"Starting evaluation with model: {model_name}")
    from evaluate import generate_critique
    
    # Process each image with each persona
    image_files = list(Path(images_dir).glob("*.png"))
    persona_files = list(Path(personas_dir).glob("*.txt"))
    
    for image_path in image_files:
        for persona_file in persona_files:
            logging.info(f"Processing {image_path.name} with {persona_file.name}")
            generate_critique(
                image_path=str(image_path),
                model_name=model_name,
                persona_file=str(persona_file)
            )
    
    logging.info("Evaluation complete")


def run_analysis(critiques_dir: str, output_dir: str) -> None:
    """Run semantic analysis on generated critiques."""
    logging.info("Starting semantic analysis...")
    from analyze import analyze_critiques
    import os
    import json
    
    # Load all critiques from directory
    critiques = []
    for critique_file in Path(critiques_dir).glob("*.txt"):
        with open(critique_file, 'r', encoding='utf-8') as f:
            critiques.append(f.read())
    
    # Analyze critiques
    results = analyze_critiques(critiques)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info("Analysis complete")


def generate_visualizations(analysis_dir: str, output_dir: str) -> None:
    """Generate visualization plots."""
    logging.info("Generating visualizations...")
    from utils import create_plots
    create_plots(analysis_dir, output_dir)
    logging.info("Visualizations complete")


def run_full_pipeline(args):
    """Execute the complete benchmark pipeline."""
    paths = setup_paths()
    
    # Phase 1: Preprocessing
    if args.skip_preprocessing:
        logging.info("Skipping preprocessing phase")
    else:
        run_preprocessing(
            args.image_dir or str(paths['data_dir'] / 'images'),
            str(paths['outputs_dir'] / 'processed_images')
        )
    
    # Phase 2: Evaluation
    if args.skip_evaluation:
        logging.info("Skipping evaluation phase")
    else:
        run_evaluation(
            str(paths['outputs_dir'] / 'processed_images'),
            args.model_name,
            str(paths['data_dir'] / 'personas')
        )
    
    # Phase 3: Analysis
    if args.skip_analysis:
        logging.info("Skipping analysis phase")
    else:
        run_analysis(
            str(paths['outputs_dir'] / 'critiques'),
            str(paths['outputs_dir'] / 'analysis')
        )
    
    # Phase 4: Visualization
    if args.skip_visualization:
        logging.info("Skipping visualization phase")
    else:
        generate_visualizations(
            str(paths['outputs_dir'] / 'analysis'),
            str(paths['outputs_dir'] / 'visualizations')
        )
    
    logging.info("Pipeline execution complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VULCA benchmark pipeline"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing phase"
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip analysis phase"
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Skip visualization phase"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run without executing"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logging.info("Dry run mode - pipeline structure verified")
        return
    
    run_full_pipeline(args)


if __name__ == "__main__":
    main()