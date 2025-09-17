"""
Tracking utilities for T-MIDAS.
"""

import os
from typing import Optional, Tuple, Any
import numpy as np
from pathlib import Path
from tmidas.utils.io_utils import read_image, write_image


def get_image_mask_pairs(folder: str, label_suffix: str) -> list[Tuple[str, str]]:
    """Get pairs of images and their corresponding masks.
    
    Args:
        folder: Directory containing images and masks
        label_suffix: Suffix for mask files
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    all_files = os.listdir(folder)
    # Only consider .tif files that do NOT end with the label_suffix
    raw_images = [f for f in all_files if f.lower().endswith('.tif') and not f.lower().endswith(label_suffix.lower())]
    
    pairs = []
    for raw_file in raw_images:
        # Remove .tif and add label_suffix
        if raw_file.lower().endswith('.tif'):
            base = raw_file[:-4]  # removes '.tif'
        else:
            base = os.path.splitext(raw_file)[0]
        label_file = f"{base}{label_suffix}"
        label_path = os.path.join(folder, label_file)
        raw_path = os.path.join(folder, raw_file)
        if os.path.exists(label_path):
            pairs.append((raw_path, label_path))
        else:
            print(f"Warning: Label file not found for {raw_file} (expected {label_file})")
    
    print(f"Found {len(pairs)} image-mask pairs.")
    return pairs


def track_cells_with_trackastra(img: np.ndarray, mask: np.ndarray, model_name: str = "general_2d", mode: str = "greedy") -> Optional[Any]:
    """Track cells using TrackAstra.
    
    Args:
        img: Image array
        mask: Mask array
        model_name: TrackAstra model name
        mode: Tracking mode
        
    Returns:
        Track graph or None if failed
    """
    try:
        import torch
        from trackastra.model import Trackastra

        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        
        # Load the model
        print(f'Loading TrackAstra model: {model_name}')
        model = Trackastra.from_pretrained(model_name, device=device)
        
        # Track
        print(f'Running tracking with mode: {mode}')
        track_graph = model.track(img, mask, mode=mode)
        
        return track_graph
    except ImportError as e:
        print(f"TrackAstra not available: {e}")
        return None


def save_tracking_results(track_graph: Any, mask: np.ndarray, output_dir: str, output_format: str = 'ctc') -> Tuple[Optional[Any], Optional[np.ndarray]]:
    """Save tracking results in specified format.
    
    Args:
        track_graph: Track graph from TrackAstra
        mask: Original mask
        output_dir: Output directory
        output_format: Output format ('ctc', 'napari', etc.)
        
    Returns:
        Tuple of (tracks, tracked_masks)
    """
    try:
        from trackastra.tracking import graph_to_ctc
        
        if output_format in ['ctc', 'both']:
            ctc_tracks, masks_tracked = graph_to_ctc(track_graph, mask, outdir=output_dir)
            return ctc_tracks, masks_tracked
    except ImportError:
        print("TrackAstra tracking functions not available")
        return None, None
