#!/usr/bin/env python
# coding: utf-8

import os
import csv
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import pyclesperanto_prototype as cle
import re
from skimage.measure import regionprops
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
Description: This script reads label images containing instance segmentations 
and regions of interest (ROIs) and counts the number of instances per ROI.
The script uses the pyclesperanto library to process the images.

The output is saved as a CSV file containing the following columns:
- ROI ID: ID of the region of interest
- instances: Number of instances per ROI
- ROI area (sq. mm): Area of the ROI in square millimeters
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Input: Folder with label images (ROIs and instance segmentations).")
    parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
    parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
    parser.add_argument("--roi_suffix", type=str, required=True, help="Suffix of the ROI label images (e.g., _ROI.tif).")
    parser.add_argument("--instance_suffix", type=str, required=True, help="Suffix of the instance segmentation label images (e.g., _labels.tif).")
    return parser.parse_args()

args = parse_args()

def get_area(ROI):
    area = cle.sum_of_all_pixels(ROI)
    area_mm2 = area * (args.pixel_resolution / 1000)**2
    return area_mm2

def counter(ROI, instances):
    instances_inside_ROI = cle.binary_and(ROI, instances)
    instances_inside_ROI = cle.connected_components_labeling_box(instances_inside_ROI)
    count = int(cle.maximum_of_all_pixels(instances_inside_ROI))
    return count

# define function that counts instances per ROI
def ROI2CSV(original_filepath, instance_filepath, roi_filepath):
    # load label images
    instances = imread(instance_filepath)
    roi_image = imread(roi_filepath)
    
    # Get unique ROI IDs (excluding background 0)
    roi_ids = np.unique(roi_image)
    roi_ids = roi_ids[roi_ids > 0]
    
    if len(roi_ids) == 0:
        print(f"Warning: No ROIs found in {roi_filepath}")
        return
    
    # create CSV file
    filename = original_filepath.replace(".tif", ".csv")
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ROI ID", "instances", "ROI area (sq. mm)"])
        
        for roi_id in roi_ids:
            # Create a binary mask for this ROI ID
            roi_mask = (roi_image == roi_id).astype(np.uint32)
            # Count instances in this ROI
            instance_count = counter(roi_mask, instances)
            # Calculate area of this ROI
            roi_area = get_area(roi_mask)
            # Write results to CSV
            writer.writerow([int(roi_id), instance_count, roi_area])

def main():
    # Find base filenames without suffixes
    instance_pattern = args.instance_suffix.replace('.', '\\.')
    roi_pattern = args.roi_suffix.replace('.', '\\.')
    
    # Get all base filenames
    all_files = os.listdir(args.input)
    instance_files = [f for f in all_files if re.search(instance_pattern + '$', f)]
    roi_files = [f for f in all_files if re.search(roi_pattern + '$', f)]
    
    # Extract base filenames
    instance_bases = [re.sub(instance_pattern + '$', '', f) for f in instance_files]
    roi_bases = [re.sub(roi_pattern + '$', '', f) for f in roi_files]
    
    # Find common base filenames
    common_bases = set(instance_bases).intersection(set(roi_bases))
    
    if not common_bases:
        print("No matching file pairs found. Please check your suffix patterns.")
        return
    
    print(f"Found {len(common_bases)} matching file pairs")
    
    for base in tqdm(common_bases, total=len(common_bases), desc="Processing images"):
        instance_file = base + args.instance_suffix
        roi_file = base + args.roi_suffix
        
        # Full paths
        instance_path = os.path.join(args.input, instance_file)
        roi_path = os.path.join(args.input, roi_file)
        original_path = os.path.join(args.input, base + ".tif")
        
        # If original file doesn't exist, use the instance file path as base for the CSV
        if not os.path.exists(original_path):
            original_path = instance_path
            
        ROI2CSV(original_path, instance_path, roi_path)

if __name__ == "__main__":
    main()