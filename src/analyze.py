#!/usr/bin/env python
"""
VULCA Framework - Analysis Module
Semantic analysis and evaluation of MLLM critiques
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance


class SemanticAnalyzer:
    """Analyze MLLM critiques using semantic embeddings."""
    
    def __init__(self, embedding_model: str = "BAAI/bge-large-zh-v1.5"):
        """
        Initialize analyzer with embedding model.
        
        Args:
            embedding_model: Name of the sentence-transformers model
        """
        self.model = SentenceTransformer(embedding_model)
        self.embeddings_cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get semantic embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = self.model.encode(text)
        return self.embeddings_cache[text]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return 1 - cosine(emb1, emb2)
    
    def calculate_emd(self, embeddings1: List[np.ndarray], embeddings2: List[np.ndarray]) -> float:
        """
        Calculate Earth Mover's Distance between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            EMD score
        """
        if not embeddings1 or not embeddings2:
            return float('inf')
        
        # Calculate pairwise distances
        distances = []
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                distances.append(cosine(emb1, emb2))
        
        # Use mean distance as simplified EMD
        return np.mean(distances)
    
    def analyze_critique_set(self, critiques: List[str]) -> Dict[str, Any]:
        """
        Analyze a set of critiques.
        
        Args:
            critiques: List of critique texts
            
        Returns:
            Analysis results dictionary
        """
        embeddings = [self.get_embedding(c) for c in critiques]
        
        # Calculate centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate diversity (mean pairwise distance)
        diversity_scores = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                diversity_scores.append(cosine(embeddings[i], embeddings[j]))
        diversity = np.mean(diversity_scores) if diversity_scores else 0
        
        # Calculate coherence (distance from centroid)
        coherence_scores = [cosine(emb, centroid) for emb in embeddings]
        coherence = 1 - np.mean(coherence_scores)
        
        return {
            'num_critiques': len(critiques),
            'diversity': diversity,
            'coherence': coherence,
            'centroid': centroid.tolist(),
            'embeddings': [emb.tolist() for emb in embeddings]
        }


def analyze_critiques(
    critiques: List[str],
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
) -> Dict[str, Any]:
    """
    Analyze a list of critiques.
    
    Args:
        critiques: List of critique texts
        embedding_model: Model for embeddings
        
    Returns:
        Analysis results
    """
    analyzer = SemanticAnalyzer(embedding_model)
    return analyzer.analyze_critique_set(critiques)


def compare_with_benchmark(
    critique: str,
    benchmark_file: str = "data/human_expert_benchmark.json",
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
) -> Dict[str, float]:
    """
    Compare a critique with human expert benchmark.
    
    Args:
        critique: Generated critique text
        benchmark_file: Path to benchmark file
        embedding_model: Model for embeddings
        
    Returns:
        Comparison metrics
    """
    analyzer = SemanticAnalyzer(embedding_model)
    
    # Load benchmark
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)
    else:
        print(f"Warning: Benchmark file not found: {benchmark_file}")
        return {}
    
    # Get embeddings
    critique_emb = analyzer.get_embedding(critique)
    benchmark_embs = [analyzer.get_embedding(b) for b in benchmark.get('critiques', [])]
    
    if not benchmark_embs:
        return {'error': 'No benchmark data available'}
    
    # Calculate metrics
    similarities = [1 - cosine(critique_emb, b_emb) for b_emb in benchmark_embs]
    
    return {
        'max_similarity': max(similarities),
        'mean_similarity': np.mean(similarities),
        'min_similarity': min(similarities),
        'std_similarity': np.std(similarities)
    }


def run_full_analysis(
    results: List[Dict[str, Any]],
    output_dir: str = "outputs/analysis"
) -> Dict[str, Any]:
    """
    Run complete analysis on evaluation results.
    
    Args:
        results: List of evaluation results
        output_dir: Directory to save analysis
        
    Returns:
        Analysis summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract critiques
    critiques = [r.get('critique', '') for r in results if r.get('critique')]
    
    if not critiques:
        return {'error': 'No critiques to analyze'}
    
    # Analyze critique set
    analyzer = SemanticAnalyzer()
    analysis = analyzer.analyze_critique_set(critiques)
    
    # Add metadata
    analysis['num_results'] = len(results)
    analysis['num_valid_critiques'] = len(critiques)
    
    # Group by persona if available
    persona_groups = {}
    for r in results:
        persona = r.get('persona', 'baseline')
        if persona not in persona_groups:
            persona_groups[persona] = []
        if r.get('critique'):
            persona_groups[persona].append(r['critique'])
    
    # Analyze each persona group
    persona_analyses = {}
    for persona, p_critiques in persona_groups.items():
        if p_critiques:
            persona_analyses[persona] = analyzer.analyze_critique_set(p_critiques)
    
    analysis['persona_analyses'] = persona_analyses
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "analysis_summary.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Analysis saved to: {analysis_file}")
    
    # Generate summary statistics
    print("\n=== Analysis Summary ===")
    print(f"Total critiques: {analysis['num_valid_critiques']}")
    print(f"Diversity score: {analysis['diversity']:.3f}")
    print(f"Coherence score: {analysis['coherence']:.3f}")
    
    if persona_analyses:
        print("\n=== Persona Analysis ===")
        for persona, p_analysis in persona_analyses.items():
            print(f"{persona}: diversity={p_analysis['diversity']:.3f}, coherence={p_analysis['coherence']:.3f}")
    
    return analysis


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MLLM critiques")
    parser.add_argument("--input", required=True, help="Input file or directory with critiques")
    parser.add_argument("--benchmark", help="Benchmark file for comparison")
    parser.add_argument("--output", default="outputs/analysis", help="Output directory")
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5", help="Embedding model")
    
    args = parser.parse_args()
    
    # Load critiques
    critiques = []
    if os.path.isfile(args.input):
        if args.input.endswith('.json'):
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    critiques = [d.get('critique', '') for d in data if d.get('critique')]
                elif isinstance(data, dict) and 'critique' in data:
                    critiques = [data['critique']]
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                critiques = [f.read()]
    elif os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.txt'):
                with open(os.path.join(args.input, file), 'r', encoding='utf-8') as f:
                    critiques.append(f.read())
    
    if not critiques:
        print("No critiques found to analyze")
        return
    
    # Analyze
    analysis = analyze_critiques(critiques, args.model)
    
    # Compare with benchmark if provided
    if args.benchmark:
        for i, critique in enumerate(critiques[:5]):  # Analyze first 5
            comparison = compare_with_benchmark(critique, args.benchmark, args.model)
            print(f"\nCritique {i+1} benchmark comparison:")
            print(f"  Mean similarity: {comparison.get('mean_similarity', 0):.3f}")
            print(f"  Max similarity: {comparison.get('max_similarity', 0):.3f}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Analysis complete. Results saved to: {output_file}")


if __name__ == "__main__":
    main()