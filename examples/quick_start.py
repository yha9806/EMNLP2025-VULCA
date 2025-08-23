#!/usr/bin/env python
"""
Quick Start Example for VULCA Framework

This script demonstrates basic usage of the VULCA framework for evaluating
MLLMs on Chinese painting critique tasks.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import vulca
sys.path.insert(0, str(Path(__file__).parent.parent))

from vulca import MLLMEvaluator


def main():
    """Main function demonstrating VULCA usage."""
    
    # Initialize evaluator
    print("Initializing VULCA Evaluator...")
    evaluator = MLLMEvaluator(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        api_endpoint="http://localhost:8000",
        api_type="vllm"
    )
    
    # Example image path (update with your actual image)
    image_path = "data/examples/january_painting.jpg"
    
    # Base prompt for Chinese painting critique
    prompt = """
    Please provide a detailed critique of this Chinese painting.
    Consider the following aspects:
    1. Composition and spatial arrangement
    2. Brushwork and technique
    3. Use of color and ink
    4. Cultural symbolism and meaning
    5. Emotional expression and artistic conception
    
    Provide your analysis in a scholarly yet accessible manner.
    """
    
    # Example persona (traditional scholar perspective)
    persona = """
    You are a traditional Chinese literati scholar from the Song Dynasty,
    well-versed in classical poetry, calligraphy, and painting theory.
    You value the expression of inner spirit (神韵) over mere technical skill,
    and appreciate subtle refinement and cultured restraint in artistic works.
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        print("Please provide a valid image path.")
        # Create a dummy example
        image_path = None
    
    if image_path:
        print(f"\nGenerating critique for: {image_path}")
        print("Using persona: Traditional Scholar")
        
        try:
            # Generate critique
            result = evaluator.generate_critique(
                image_path=image_path,
                prompt=prompt,
                persona=persona,
                max_tokens=1024,
                temperature=0.7
            )
            
            # Display results
            print("\n" + "="*60)
            print("GENERATED CRITIQUE:")
            print("="*60)
            print(result["critique"])
            print("\n" + "="*60)
            print(f"Model: {result['model']}")
            print(f"Temperature: {result['temperature']}")
            print(f"Max Tokens: {result['max_tokens']}")
            
        except Exception as e:
            print(f"Error generating critique: {e}")
            print("\nMake sure:")
            print("1. vLLM server is running on port 8000")
            print("2. The model is loaded correctly")
            print("3. The image path is valid")
    else:
        print("\nDemo mode - no actual critique generated.")
        print("To run with real data:")
        print("1. Start vLLM server:")
        print("   python -m vllm.entrypoints.openai.api_server \\")
        print("     --model Qwen/Qwen2.5-VL-7B-Instruct \\")
        print("     --port 8000 --trust-remote-code")
        print("2. Place your image in data/examples/")
        print("3. Update the image_path variable")
        print("4. Run this script again")


if __name__ == "__main__":
    main()