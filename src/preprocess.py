#!/usr/bin/env python
"""
VULCA Framework - Image Preprocessing Module
Adaptive sliding window implementation for painting analysis
Simplified version based on noAI1.5.py
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage.feature import local_binary_pattern


def calculate_saliency(image: np.ndarray) -> float:
    """
    Calculate saliency score using edge detection and texture analysis.
    
    Args:
        image: Input image array
        
    Returns:
        Saliency score between 0 and 1
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Edge detection using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(edge_magnitude) / 255.0
    
    # Texture analysis using LBP
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    texture_complexity = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-6))
    
    # Combine edge and texture features
    saliency = 0.6 * edge_density + 0.4 * (texture_complexity / 4.0)
    return min(max(saliency, 0.0), 1.0)


def adaptive_sliding_window(
    image: np.ndarray,
    window_sizes: List[int] = [2560, 1280, 640],
    output_size: int = 640,
    saliency_threshold_low: float = 0.25,
    saliency_threshold_high: float = 0.6
) -> List[np.ndarray]:
    """
    Apply adaptive sliding window to extract image patches.
    
    Args:
        image: Input image
        window_sizes: List of window sizes to try
        output_size: Output size for all patches
        saliency_threshold_low: Low saliency threshold
        saliency_threshold_high: High saliency threshold
        
    Returns:
        List of extracted and resized patches
    """
    height, width = image.shape[:2]
    patches = []
    
    for window_size in window_sizes:
        # Skip if window is larger than image
        if window_size > min(height, width):
            continue
        
        # Calculate stride based on window size
        base_stride = int(window_size * 0.75)  # Default 75% overlap
        
        # Sliding window
        for y in range(0, height - window_size + 1, base_stride):
            for x in range(0, width - window_size + 1, base_stride):
                # Extract patch
                patch = image[y:y+window_size, x:x+window_size]
                
                # Calculate saliency
                saliency = calculate_saliency(patch)
                
                # Adaptive stride based on saliency
                if saliency < saliency_threshold_low:
                    # Low information, larger stride
                    stride_factor = 0.9
                elif saliency > saliency_threshold_high:
                    # High information, smaller stride
                    stride_factor = 0.5
                else:
                    # Medium information
                    stride_factor = 0.75
                
                # Resize to output size
                resized = cv2.resize(patch, (output_size, output_size))
                patches.append(resized)
    
    return patches


def process_image(
    image_path: str,
    output_dir: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Process a single image with adaptive sliding window.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save processed patches
        **kwargs: Additional parameters for adaptive_sliding_window
        
    Returns:
        List of paths to saved patches (or patches if no output_dir)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Apply adaptive sliding window
    patches = adaptive_sliding_window(image, **kwargs)
    
    # Save or return patches
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        saved_paths = []
        for i, patch in enumerate(patches):
            output_path = os.path.join(output_dir, f"{base_name}_patch_{i:04d}.jpg")
            cv2.imwrite(output_path, patch)
            saved_paths.append(output_path)
        
        print(f"✓ Processed {len(patches)} patches from {image_path}")
        return saved_paths
    else:
        return patches


def process_directory(
    input_dir: str,
    output_dir: str,
    **kwargs
) -> int:
    """
    Process all images in a directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed patches
        **kwargs: Additional parameters for processing
        
    Returns:
        Total number of patches processed
    """
    total_patches = 0
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            patches = process_image(image_path, output_dir, **kwargs)
            total_patches += len(patches)
        except Exception as e:
            print(f"✗ Error processing {image_file}: {e}")
    
    print(f"\n✓ Processing complete: {total_patches} total patches")
    return total_patches


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive sliding window preprocessing")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", default="outputs/patches", help="Output directory")
    parser.add_argument("--window-sizes", nargs='+', type=int, default=[2560, 1280, 640],
                       help="Window sizes to use")
    parser.add_argument("--output-size", type=int, default=640,
                       help="Output patch size")
    parser.add_argument("--threshold-low", type=float, default=0.25,
                       help="Low saliency threshold")
    parser.add_argument("--threshold-high", type=float, default=0.6,
                       help="High saliency threshold")
    
    args = parser.parse_args()
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        process_image(
            args.input,
            args.output,
            window_sizes=args.window_sizes,
            output_size=args.output_size,
            saliency_threshold_low=args.threshold_low,
            saliency_threshold_high=args.threshold_high
        )
    elif os.path.isdir(args.input):
        process_directory(
            args.input,
            args.output,
            window_sizes=args.window_sizes,
            output_size=args.output_size,
            saliency_threshold_low=args.threshold_low,
            saliency_threshold_high=args.threshold_high
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()