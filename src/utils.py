#!/usr/bin/env python
"""
VULCA Framework - Utility Functions
Common utilities and helper functions
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save data to YAML file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)


def create_plots(analysis_dir: str, output_dir: str) -> None:
    """
    Create visualization plots from analysis results.
    
    Args:
        analysis_dir: Directory containing analysis results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Placeholder for plotting logic
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.text(0.5, 0.5, 'VULCA Framework Visualization', 
            ha='center', va='center', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    output_path = Path(output_dir) / 'sample_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def ensure_directories(*dirs: str) -> None:
    """Ensure directories exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)