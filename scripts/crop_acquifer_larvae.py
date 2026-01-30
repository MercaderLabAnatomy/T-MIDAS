import os
import numpy as np
import argparse
import tifffile as tf
from skimage.measure import regionprops, label
import re
from collections import defaultdict
from tqdm import tqdm
import pyclesperanto_prototype as cle


"""
This script extracts elongated ROIs from Acquifer TIF files and creates z-stacks 
by grouping files from the same well, position, and channel.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract elongated ROIs from Acquifer TIF files and save them as multi-color z-stack TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the TIF files.')
    parser.add_argument('--padding', type=int, default=50, help='Padding around the ROI (default: 50)')
    return parser.parse_args()

def parse_acquifer_filename(filename):
    """
    Parse Acquifer filename to extract metadata.
    Example: -A001--PO01--LO001--CO1--SL001--PX16250--PW0040--IN0020--TM235--X014398--Y011017--Z211182--T0000000000--WE00001.tif
    """
    pattern = r'^-([A-Z]\d+)--PO(\d+)--LO(\d+)--CO(\d+)--SL(\d+)'
    match = re.match(pattern, filename)
    if match:
        return {
            'well': match.group(1),      # A001
            'position': match.group(2),   # 01
            'location': match.group(3),   # 001
            'channel': match.group(4),    # 1 (CO1 = channel 1)
            'slice': match.group(5)       # 001 (SL001 = slice 1)
        }
    return None

def group_files_by_stack(input_folder):
    """
    Group files by well, position, location, and channel to create z-stacks.
    Each group will contain all z-slices for the same imaging position and channel.
    """
    files_grouped_by_stack = defaultdict(list)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            metadata = parse_acquifer_filename(filename)
            if metadata:
                # Create a key that uniquely identifies a z-stack
                # (same well, position, location, channel but different slices)
                stack_key = f"{metadata['well']}_PO{metadata['position']}_LO{metadata['location']}_CO{metadata['channel']}"
                files_grouped_by_stack[stack_key].append({
                    'filename': filename,
                    'slice': int(metadata['slice']),
                    'metadata': metadata
                })
    
    # Sort files within each group by slice number
    for stack_key in files_grouped_by_stack:
        files_grouped_by_stack[stack_key].sort(key=lambda x: x['slice'])
    
    return files_grouped_by_stack

def get_roi_from_brightfield(input_folder, stack_groups, padding):
    """
    Get ROI from the brightfield channel (CO1) for each well/position/location combination.
    """
    roi_dict = {}
    
    # Find all unique well/position/location combinations
    locations = set()
    for stack_key in stack_groups:
        parts = stack_key.split('_')
        location_key = f"{parts[0]}_{parts[1]}_{parts[2]}"  # well_position_location
        locations.add(location_key)
    
    for location_key in locations:
        # Look for CO1 (brightfield) stack for this location
        co1_stack_key = f"{location_key}_CO1"
        
        if co1_stack_key in stack_groups:
            # Use the middle slice of the CO1 stack to determine ROI
            co1_files = stack_groups[co1_stack_key]
            middle_idx = len(co1_files) // 2
            middle_file = co1_files[middle_idx]['filename']
            
            # Load the middle slice and calculate ROI
            with tf.TiffFile(os.path.join(input_folder, middle_file)) as tif:
                co1_image = tif.asarray()
            roi = get_roi(co1_image, padding)
            roi_dict[location_key] = roi
            print(f"Calculated ROI for {location_key}: {roi}")
        else:
            print(f"Warning: No CO1 (brightfield) stack found for {location_key}")
    
    return roi_dict

def get_roi(image, padding):
    """Calculate ROI from a single image."""
    if image is None or image.size == 0:
        return (0, 0, 0, 0)  # Return a default ROI if the image is None or empty

    # Normalize and convert to uint8
    image_min, image_max = np.min(image), np.max(image)
    if image_max > image_min:
        image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)  # All black image if min == max

    image1_gbp = cle.gaussian_blur(image, None, 10.0, 10.0, 0.0)
    image3_vsp = cle.variance_sphere(image1_gbp, None, 1.0, 1.0, 0.0)
    labels = cle.threshold_otsu(image3_vsp)

    # Get ROIs
    props = regionprops(labels, intensity_image=image)
    if len(props) == 0:
        # If no objects found, return the whole image
        return (0, 0, image.shape[0], image.shape[1])
    
    y0, x0, y1, x1 = props[0].bbox
    minr, minc = max(0, y0 - padding), max(0, x0 - padding)
    maxr, maxc = min(labels.shape[0], y1 + padding), min(labels.shape[1], x1 + padding)
    
    return (minr, minc, maxr - minr, maxc - minc)

def create_zstack(input_folder, file_list, roi):
    """
    Create a z-stack from a list of files and crop using the provided ROI.
    """
    z_slices = []
    y, x, h, w = roi
    
    for file_info in file_list:
        filename = file_info['filename']
        with tf.TiffFile(os.path.join(input_folder, filename)) as tif:
            image = tif.asarray()
        
        # Crop the image
        cropped_image = image[y:y+h, x:x+w]
        
        # Normalize to 16-bit
        normalized_image = ((cropped_image - cropped_image.min()) / 
                          (cropped_image.max() - cropped_image.min()) * 65535).astype(np.uint16)
        
        z_slices.append(normalized_image)
    
    # Stack all z-slices
    z_stack = np.stack(z_slices, axis=0)
    return z_stack

def process_stacks(input_folder, output_dir, padding):
    """Process all z-stacks."""
    # Group files by stack
    stack_groups = group_files_by_stack(input_folder)
    print(f"Found {len(stack_groups)} z-stacks to process.")
    
    # Get ROIs from brightfield images
    roi_dict = get_roi_from_brightfield(input_folder, stack_groups, padding)
    
    # Process each z-stack
    for stack_key, file_list in tqdm(stack_groups.items(), desc="Processing z-stacks"):
        try:
            # Extract location key to get corresponding ROI
            parts = stack_key.split('_')
            location_key = f"{parts[0]}_{parts[1]}_{parts[2]}"  # well_position_location
            
            if location_key not in roi_dict:
                print(f"Warning: No ROI found for {stack_key}. Skipping.")
                continue
            
            roi = roi_dict[location_key]
            
            # Create z-stack
            z_stack = create_zstack(input_folder, file_list, roi)
            
            # Generate output filename
            # Format: well_position_location_channel_zstack.tif
            output_filename = f"{stack_key}_cropped_zstack.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save z-stack with ImageJ metadata
            metadata = {
                'ImageJ': '1.53c',
                'images': z_stack.shape[0],
                'slices': z_stack.shape[0],
                'hyperstack': True,
                'mode': 'grayscale',
                'unit': 'pixel'
            }
            
            tf.imwrite(output_path, z_stack, imagej=True, metadata=metadata, compression='zlib')
            print(f"Saved z-stack: {output_filename} (shape: {z_stack.shape})")
            
        except Exception as e:
            print(f"Error processing stack {stack_key}: {str(e)}")

def main():
    args = parse_args()
    input_folder = args.input
    padding = args.padding
    output_dir = os.path.join(input_folder, "processed_zstacks")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_stacks(input_folder, output_dir, padding)
    print(f"\nProcessing complete. Z-stacks saved to: {output_dir}")

if __name__ == "__main__":
    main()