#!/usr/bin/env python3
"""
TrackAstra Cell Tracking Module for T-MIDAS

This script uses TrackAstra to track cells in time-lapse microscopy images.
It processes image pairs sequentially without storing them all in memory,
and outputs tracking results in various formats.

The implementation automatically sets up a dedicated conda environment
for TrackAstra if it doesn't exist.
"""

import argparse
import os
import sys
import glob
import re
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread
from tifffile import imwrite

# Model descriptions from TrackAstra repository
MODEL_DESCRIPTIONS = {
    "general_2d": "For tracking fluorescent nuclei, bacteria (PhC), whole cells (BF, PhC, DIC), epithelial cells with fluorescent membrane, budding yeast cells (PhC), fluorescent particles, etc.",
    "ctc (2D,3D)": "For tracking Cell Tracking Challenge datasets. This is the successor of the winning model of the ISBI 2024 CTC generalizable linking challenge."
}

def get_image_mask_pairs(folder, label_suffix):
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




def check_trackastra_env():
    """Check if trackastra environment exists, create it if it doesn't."""
    result = subprocess.run("mamba env list", shell=True, capture_output=True, text=True)
    if "trackastra" not in result.stdout:
        print("TrackAstra environment not found. Creating it now...")
        
        # Create commands to set up the environment
        commands = [
            "mamba create --name trackastra python=3.10 -y",
            "mamba install -n trackastra -c conda-forge -c gurobi -c funkelab ilpy -y",
            "mamba install -n trackastra -c conda-forge scikit-image numpy matplotlib tqdm tifffile -y",
            "mamba run -n trackastra pip install trackastra[ilp]"
        ]
        
        # Execute each command
        for cmd in commands:
            print(f"Executing: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        print("TrackAstra environment created successfully.")
    else:
        print("TrackAstra environment found.")

def process_data(args):
    """
    Process tracking using TrackAstra sequentially.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Print available models
    print("\nAvailable TrackAstra models:")
    print("-" * 50)
    for model, description in MODEL_DESCRIPTIONS.items():
        print(f"* {model}: {description}")
    print("-" * 50)
    
    # Create output tracking directory if needed
    if args.output_format in ['ctc', 'both']:
        tracking_dir = os.path.join(args.input, "tracked")
        os.makedirs(tracking_dir, exist_ok=True)
    
    # Get image-mask pairs
    pairs = get_image_mask_pairs(args.input, args.label_suffix)
    if not pairs:
        print("Error: No valid image-mask pairs found. Exiting.")
        return 1
    
    # Check if TrackAstra environment exists
    check_trackastra_env()
    
    # Process pairs sequentially in the trackastra environment
    print(f"Processing {len(pairs)} image-mask pairs sequentially...")
    
    # Run the tracking in the trackastra environment 
    # Process each pair one at a time
    for i, (img_file, mask_file) in enumerate(tqdm(pairs, desc="Processing pairs")):
        print(f"\nProcessing pair {i+1}/{len(pairs)}: {os.path.basename(img_file)} and {os.path.basename(mask_file)}")
        
        # Prepare a command to process a single pair
        cmd = f"""
        mamba run -n trackastra python -c "
import os
import sys
import numpy as np
from skimage.io import imread
from tifffile import imwrite

try:
    # Import TrackAstra
    import torch
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {{device}}')
    
    # Load a single image-mask pair
    print('Loading image and mask...')
    
    img, mask = imread('{img_file}'), imread('{mask_file}')
    
    print(f'Loaded image with shape {{img.shape}} and mask with shape {{mask.shape}}')
    
    # Load the model (done for each pair to ensure clean state)
    print('Loading TrackAstra model: {args.model}')
    model = Trackastra.from_pretrained('{args.model}', device=device)
    
    # Track a single pair
    print('Running tracking with mode: {args.mode}')
    track_graph = model.track(img, mask, mode='{args.mode}')
    
    # Save results
    if '{args.output_format}' in ['ctc', 'both']:
        print('Generating tracking data...')
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            mask,
            outdir='{tracking_dir if args.output_format in ["ctc", "both"] else None}'
        )
        
        # Save the tracked mask with _tracked.tif suffix
        if '{args.output_format}' in ['tif', 'both']:
            output_file = '{mask_file}'.replace('{args.label_suffix}', '_tracked.tif')
            imwrite(output_file, masks_tracked, compression='zlib')
            print(f'Saved tracked mask: {{output_file}}')
    
    elif '{args.output_format}' == 'tif':
        # If only tif format is requested without CTC format
        print('Generating tracking data...')
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            mask,
            outdir=None  # Don't write CTC files, just get the tracked masks
        )
        
        output_file = '{mask_file}'.replace('{args.label_suffix}', '_tracked.tif')
        imwrite(output_file, masks_tracked, compression='zlib')
        print(f'Saved tracked mask: {{output_file}}')
    
    if '{args.output_format}' in ['napari', 'both']:
        print('Generating napari tracks format...')
        napari_tracks = graph_to_napari_tracks(track_graph)
        napari_tracks_path = os.path.join('{args.input}', 'napari_tracks_pair_{0}.npy'.format({i+1}))
        np.save(napari_tracks_path, napari_tracks)
        print(f'Saved napari tracks: {{napari_tracks_path}}')
    
    print('Tracking for this pair completed successfully!')
    sys.exit(0)

except Exception as e:
    print(f'Error during tracking: {{str(e)}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
        """
        
        # Execute the command for this pair
        result = subprocess.run(cmd, shell=True, text=True)
        
        if result.returncode != 0:
            print(f"Error processing pair {i+1}/{len(pairs)}: {os.path.basename(img_file)} and {os.path.basename(mask_file)}")
        
        # Clean up GPU memory after each pair
        try:
            subprocess.run("nvidia-smi -r", shell=True, capture_output=True)
        except:
            # If nvidia-smi is not available or fails, just continue
            pass
    

    
    print("\nAll pairs processed successfully!")
    return 0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track cells using TrackAstra.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing both time series images and their masks.")
    parser.add_argument("--model", type=str, default="general_2d", choices=["general_2d", "ctc"],
                        help="TrackAstra model to use for tracking.")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "ilp", "greedy_nodiv"],
                        help="Tracking mode: greedy (faster), ilp (more accurate, handles divisions), greedy_nodiv (like greedy but without divisions).")
    parser.add_argument("--label_suffix", type=str, default="_labels.tif", help="Suffix identifying label images.")
    parser.add_argument("--time_dimension", type=int, default=0, help="Time dimension in the image stack (0-based).")
    parser.add_argument("--output_format", type=str, default="both", choices=["ctc", "napari", "both", "tif"],
                        help="Format for the output tracking results: ctc (Cell Tracking Challenge), napari, tif (tracked mask images), or both.")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Process data
    return process_data(args)

if __name__ == "__main__":
    sys.exit(main())