#!/usr/bin/env python
"""
VULCA Framework - Unified Interface
Vision-Understanding and Language-based Cultural Adaptability Framework
"""

import os
import json
import yaml
from typing import Dict, Optional, List, Any
from .evaluate import generate_critique
from .preprocess import process_image
from .analyze import analyze_critiques


class VULCA:
    """
    Unified interface for the VULCA framework.
    Provides a simple API for painting evaluation using MLLMs with cultural personas.
    """
    
    def __init__(self, config_path: str = "configs/default.yaml"):
        """
        Initialize VULCA framework with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.personas = self._load_personas()
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'model': {
                    'name': 'Qwen/Qwen2.5-VL-7B-Instruct',
                    'api_endpoint': 'http://localhost:8000/v1/chat/completions',
                    'max_tokens': 2048,
                    'temperature': 0.7
                },
                'preprocessing': {
                    'window_sizes': [2560, 1280, 640],
                    'output_size': 640,
                    'saliency_threshold_low': 0.25,
                    'saliency_threshold_high': 0.6
                },
                'analysis': {
                    'embedding_model': 'BAAI/bge-large-zh-v1.5',
                    'embedding_dim': 1024
                }
            }
    
    def _load_personas(self) -> Dict[str, str]:
        """Load persona definitions from markdown files."""
        personas = {}
        persona_dir = "data/personas"
        
        if os.path.exists(persona_dir):
            for file in os.listdir(persona_dir):
                if file.endswith('.md'):
                    persona_name = file.replace('.md', '')
                    with open(os.path.join(persona_dir, file), 'r', encoding='utf-8') as f:
                        personas[persona_name] = f.read()
        
        return personas
    
    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base from JSON file."""
        kb_path = "data/knowledge/knowledge_base.json"
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def evaluate_painting(
        self,
        image_path: str,
        persona: Optional[str] = None,
        output_dir: str = "outputs/critiques"
    ) -> Dict[str, Any]:
        """
        Evaluate a painting using MLLM with optional persona enhancement.
        
        Args:
            image_path: Path to the painting image
            persona: Name of cultural persona to use (optional)
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing critique and analysis results
        """
        results = {
            'image_path': image_path,
            'persona': persona,
            'timestamp': None,
            'critique': None,
            'analysis': None,
            'error': None
        }
        
        try:
            # Step 1: Preprocess image if needed
            if self.config.get('preprocessing', {}).get('enabled', False):
                processed_images = process_image(
                    image_path,
                    **self.config['preprocessing']
                )
                # Use first slice for evaluation (simplified)
                eval_image = processed_images[0] if processed_images else image_path
            else:
                eval_image = image_path
            
            # Step 2: Generate critique
            persona_text = self.personas.get(persona, "") if persona else ""
            critique = generate_critique(
                image_path=eval_image,
                model_name=self.config['model']['name'],
                persona_text=persona_text,
                knowledge_base=self.knowledge_base,
                api_endpoint=self.config['model']['api_endpoint'],
                model_params={
                    'max_tokens': self.config['model']['max_tokens'],
                    'temperature': self.config['model']['temperature']
                },
                output_dir=output_dir
            )
            results['critique'] = critique
            
            # Step 3: Analyze critique (optional)
            if self.config.get('analysis', {}).get('enabled', False):
                analysis = analyze_critiques(
                    [critique],
                    embedding_model=self.config['analysis']['embedding_model']
                )
                results['analysis'] = analysis
            
            # Save results
            from datetime import datetime
            results['timestamp'] = datetime.now().isoformat()
            
            output_path = os.path.join(
                output_dir,
                f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Evaluation complete. Results saved to: {output_path}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"✗ Evaluation failed: {e}")
        
        return results
    
    def batch_evaluate(
        self,
        image_dir: str,
        personas: Optional[List[str]] = None,
        output_dir: str = "outputs/batch"
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple paintings with optional multiple personas.
        
        Args:
            image_dir: Directory containing painting images
            personas: List of personas to use (None for baseline)
            output_dir: Directory to save outputs
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Get all image files
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        # If no personas specified, use baseline (None)
        if personas is None:
            personas = [None]
        
        # Process each image with each persona
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            for persona in personas:
                print(f"\nEvaluating: {image_file} with persona: {persona or 'baseline'}")
                result = self.evaluate_painting(image_path, persona, output_dir)
                result['image_file'] = image_file
                results.append(result)
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Batch evaluation complete. {len(results)} evaluations saved.")
        return results
    
    def run_experiment(self, experiment_config: str = "configs/hyperparams.yaml"):
        """
        Run a complete experiment based on configuration file.
        
        Args:
            experiment_config: Path to experiment configuration
        """
        with open(experiment_config, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)
        
        print("Starting VULCA experiment...")
        print(f"Configuration: {experiment_config}")
        
        # Run batch evaluation
        results = self.batch_evaluate(
            image_dir=exp_config['data']['image_dir'],
            personas=exp_config.get('personas', None),
            output_dir=exp_config['output']['dir']
        )
        
        # Run analysis if enabled
        if exp_config.get('analysis', {}).get('enabled', False):
            from .analyze import run_full_analysis
            analysis_results = run_full_analysis(
                results,
                output_dir=exp_config['output']['dir']
            )
            print(f"✓ Analysis complete. Results in: {exp_config['output']['dir']}")
        
        print("\n✓ Experiment complete!")
        return results


def main():
    """Command-line interface for VULCA framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VULCA Framework CLI')
    parser.add_argument('--image', help='Path to painting image')
    parser.add_argument('--persona', help='Cultural persona to use')
    parser.add_argument('--config', default='configs/default.yaml', help='Configuration file')
    parser.add_argument('--batch', help='Directory for batch processing')
    parser.add_argument('--experiment', help='Run full experiment from config')
    
    args = parser.parse_args()
    
    # Initialize VULCA
    vulca = VULCA(config_path=args.config)
    
    if args.experiment:
        # Run full experiment
        vulca.run_experiment(args.experiment)
    elif args.batch:
        # Batch processing
        personas = [args.persona] if args.persona else None
        vulca.batch_evaluate(args.batch, personas)
    elif args.image:
        # Single image evaluation
        result = vulca.evaluate_painting(args.image, args.persona)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()