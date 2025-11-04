"""
ROI Colocalization Analysis for 2 or 3 Color Channels

This script performs colocalization analysis between 2 or 3 channels:
- Channel 1: ROI labels (required)
- Channel 2: Object labels (required)
- Channel 3: Object labels OR intensity image (optional)

When using 3 channels:
1. Label-based mode (--channel3_is_labels y): Count Channel 3 objects within Channel 2 and Channel 1
2. Intensity-based mode (--channel3_is_labels n): Measure Channel 3 intensity statistics within regions

Example Usage:

# 2 channels (labels only)
python ROI_colocalization_count_multicolor.py --input /data \
    --channels CellMask DAPI \
    --label_patterns "*_labels.tif" "*_labels.tif" \
    --get_sizes y --size_method median

# 3 channels (all labels)
python ROI_colocalization_count_multicolor.py --input /data \
    --channels CellMask DAPI EdU \
    --label_patterns "*_labels.tif" "*_labels.tif" "*_labels.tif" \
    --channel3_is_labels y \
    --get_sizes y --size_method median

# 3 channels (channel 3 as intensity)
python ROI_colocalization_count_multicolor.py --input /data \
    --channels CellMask DAPI GFP \
    --label_patterns "*_labels.tif" "*_labels.tif" "*.tif" \
    --channel3_is_labels n \
    --get_sizes y

# 3 channels with positive object counting (e.g., KI67+ nuclei)
python ROI_colocalization_count_multicolor.py --input /data \
    --channels GFP DAPI KI67 \
    --label_patterns "*_labels.tif" "*_labels.tif" "*.tif" \
    --channel3_is_labels n \
    --count_positive y \
    --threshold_method percentile \
    --threshold_value 75.0

# Using wildcards for varying label suffixes (e.g., _labels1, _labels23, _labels5)
python ROI_colocalization_count_multicolor.py --input /data \
    --channels ROI Nuclei Markers \
    --label_patterns "*_labels[0-9]*.tif" "*_labels[0-9]*.tif" "*.tif" \
    --channel3_is_labels n

# Match files with single digit suffix only
python ROI_colocalization_count_multicolor.py --input /data \
    --channels C1 C2 C3 \
    --label_patterns "*_labels?.tif" "*_nuclei.tif" "*.tif"
"""

import os
from difflib import SequenceMatcher
from collections import defaultdict
import glob
import csv
import argparse
import traceback
import gc
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from skimage.transform import resize

try:
    import cupy as cp
    import cucim.skimage as cusk
    from skimage import io, measure

    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    from skimage import io, measure

    GPU_AVAILABLE = False

# Constants
LARGE_FILE_THRESHOLD_BYTES = 1e9  # 1 GB
DEFAULT_BATCH_SIZE = 10
MIN_CHANNELS = 2
MAX_CHANNELS = 3

@contextmanager
def memory_manager():
    """Context manager to handle memory cleanup after operations."""
    try:
        yield
    finally:
        gc.collect()
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

def parse_args():
    """Parse command line arguments with better organization."""
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    
    # Input/output arguments
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the parent folder of the channel folders.')
    
    # Channel arguments
    parser.add_argument('--channels', nargs='+', type=str, required=True, 
                         help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True,
                         help='Glob pattern for each channel. Supports wildcards: * (any chars), ? (one char), [seq] (char in seq). '
                              'Examples: "*_labels.tif" matches any prefix with _labels.tif; '
                              '"*_labels[0-9]*.tif" matches _labels1, _labels23, etc.; '
                              '"*_labels?.tif" matches single digit like _labels5.tif')
    parser.add_argument('--channel3_is_labels', type=str, default='y',
                         help='Is channel 3 a label image? (y/n). If "n", will measure intensity statistics instead of counting objects. Only applies when using 3 channels.')
    
    # Threshold options for positive object counting
    parser.add_argument('--count_positive', type=str, default='n',
                         help='Count Channel 2 objects positive for Channel 3 signal? (y/n). Only applies when channel3_is_labels=n.')
    parser.add_argument('--threshold_method', type=str, choices=['percentile', 'absolute'], default='percentile',
                         help='Method for determining positive threshold: "percentile" or "absolute"')
    parser.add_argument('--threshold_value', type=float, default=75.0,
                         help='Threshold value. For percentile: 0-100 (e.g., 75 for 75th percentile). For absolute: intensity value.')
    
    # Analysis options
    parser.add_argument('--get_sizes', type=str, default='n', 
                          help='Do you want to get sizes of ROIs in all channels? (y/n)')
    parser.add_argument('--size_method', type=str, choices=['median', 'sum'],
                          help='Method to calculate sizes for second and third channels')
    
    # Performance options
    parser.add_argument('--num_workers', type=int, default=1, 
                      help='Number of worker processes to use when GPU is not available')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, 
                      help='Number of images to process in each batch')
    parser.add_argument('--no_resize', action='store_true', 
                      help='Disable image resizing. Images must be the same size if this is used.')
    
    args = parser.parse_args()
    
    # Validate size_method if get_sizes is 'y'
    if args.get_sizes.lower() == 'y' and args.size_method is None:
        args.size_method = input("\nWhich size stats? Type median or sum: ")
        while args.size_method not in ['median', 'sum']:
            print("Invalid input. Please enter 'median' or 'sum'.")
            args.size_method = input("\nWhich size stats? Type median or sum: ")
    
    return args

# def validate_file_lists(file_lists, channels):
#     """Ensures all channels have the same number of files and they exist."""
#     lengths = [len(file_lists[channel]) for channel in channels]
#     if len(set(lengths)) != 1:
#         raise ValueError(f"Channel folders contain different numbers of files: {lengths}")

#     # Check file existence
#     for channel in channels:
#         for filepath in file_lists[channel]:
#             if not os.path.exists(filepath):
#                 raise FileNotFoundError(f"File not found: {filepath}")

#     return True



def longest_common_substring(s1, s2):
    """Finds the longest common substring between two strings."""
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size]

def group_files_by_common_substring(file_lists, channels):
    """
    Groups files across channels based on the longest common substring in their filenames.

    Args:
        file_lists (dict): A dictionary where keys are channel names and values are lists of file paths.
        channels (list): A list of channel names corresponding to the keys in file_lists.

    Returns:
        dict: A dictionary where keys are common substrings and values are lists of file paths grouped by substring.
    """
    all_files = [os.path.basename(file) for channel in channels for file in file_lists[channel]]
    groups = defaultdict(list)

    while all_files:
        current = all_files.pop(0)
        group = [current]
        to_remove = []

        for other in all_files:
            common_substring = longest_common_substring(current, other)
            if len(common_substring) > 1:  # Only consider substrings of length > 1
                group.append(other)
                to_remove.append(other)

        for item in to_remove:
            all_files.remove(item)

        # Use the longest substring as the key for this group
        common_key = current
        for file in group:
            common_key = longest_common_substring(common_key, file)

        groups[common_key].extend(group)

    return groups



def validate_file_lists(file_lists, channels):
    """Ensures all channels have the same number of files, they exist, and identifies files without common substrings."""
    # Check file existence
    for channel in channels:
        for filepath in file_lists[channel]:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

    # Group files by common substrings
    grouped_files = group_files_by_common_substring(file_lists, channels)

    # Identify files without common substrings across all channels
    files_without_common_substring = []
    for channel in channels:
        for file in file_lists[channel]:
            basename = os.path.basename(file)
            if not any(basename in group for group in grouped_files.values()):
                files_without_common_substring.append(basename)

    if files_without_common_substring:
        print("Files without common substrings across all folders:")
        for file in set(files_without_common_substring):
            print(f"- {file}")

    # Check if all channels have the same number of files
    lengths = [len(file_lists[channel]) for channel in channels]
    if len(set(lengths)) != 1:
        raise ValueError(f"Channel folders contain different numbers of files: {lengths}")

    return True






def load_image(file_path, output_shape=None):
    """Load an image file with proper error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Check file size before loading
        file_size = os.path.getsize(file_path)
        if file_size > LARGE_FILE_THRESHOLD_BYTES:
            print(f"Warning: Large file detected ({file_size/1e6:.1f} MB): {file_path}")

        image = io.imread(file_path)
        if image is None or image.size == 0:
            raise ValueError(f"Failed to load image or empty image: {file_path}")

        # Resize if needed
        if output_shape is not None and image.shape != output_shape:
            image = resize(image, output_shape, anti_aliasing=True, preserve_range=True)
            image = np.round(image).astype(image.dtype)  # Round to nearest integer and convert to original dtype

        # Convert to GPU array if available
        if GPU_AVAILABLE:
            return cp.asarray(image)
        else:
            return image
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        raise

def safe_median(arr):
    """Safely calculate median with GPU support."""
    if len(arr) == 0:
        return 0
    if GPU_AVAILABLE:
        return cp.median(arr).get()
    else:
        return np.median(arr)

def safe_sum(arr):
    """Safely calculate sum with GPU support."""
    if len(arr) == 0:
        return 0
    if GPU_AVAILABLE:
        return cp.sum(arr).get()
    else:
        return np.sum(arr)

def get_nonzero_labels(image):
    """Get unique, non-zero labels from an image."""
    xp = cp if GPU_AVAILABLE else np
    mask = image != 0
    labels = xp.unique(image[mask])
    
    # Convert to Python integers
    if GPU_AVAILABLE:
        return [int(x) for x in labels.get()]
    else:
        return [int(x) for x in labels]

def count_unique_nonzero(array, mask):
    """Count unique non-zero values in array where mask is True."""
    xp = cp if GPU_AVAILABLE else np
    unique_vals = xp.unique(array[mask])
    count = len(unique_vals)
    
    # Remove 0 from count if present
    if count > 0:
        if GPU_AVAILABLE:
            if 0 in unique_vals.get():
                count -= 1
        else:
            if 0 in unique_vals:
                count -= 1
            
    return count

def calculate_all_rois_size(image):
    """
    Calculate sizes of all ROIs in the given image.

    Args:
        image: Labeled image

    Returns:
        dict: Dictionary mapping label IDs to their sizes
    """
    xp = cp if GPU_AVAILABLE else np
    measure_func = cusk.measure if GPU_AVAILABLE else measure
    sizes = {}

    try:
        # Convert to int32 to avoid potential overflow issues with regionprops
        image_int = image.astype(xp.uint32)

        for prop in measure_func.regionprops(image_int):
            label = int(prop.label)  # Ensure label is a standard Python integer
            size_value = float(prop.area) if hasattr(prop.area, 'item') else prop.area
            sizes[label] = int(size_value)  # Convert to Python integer
    except Exception as e:
        print(f"Error calculating ROI sizes: {str(e)}")
        traceback.print_exc()

    return sizes

def calculate_coloc_size(image_c1, image_c2, label_id, mask_c2=None, image_c3=None):
    """
    Calculate the size of colocalization between channels.

    Args:
        image_c1: First channel image with ROI labels
        image_c2: Second channel image
        label_id: Label ID in image_c1 to analyze
        mask_c2: Boolean flag indicating whether to include (True) or exclude (False) image_c2
        image_c3: Optional third channel image

    Returns:
        int: size of colocalization
    """
    xp = cp if GPU_AVAILABLE else np

    # Create mask for current ROI
    mask = (image_c1 == int(label_id))

    # Handle mask_c2 parameter
    if mask_c2 is not None:
        if mask_c2:
            # sizes where c2 is present
            mask = mask & (image_c2 != 0)
            target_image = image_c3 if image_c3 is not None else image_c2
        else:
            # sizes where c2 is NOT present
            mask = mask & (image_c2 == 0)
            if image_c3 is None:
                # If no image_c3, just return count of mask pixels
                return xp.count_nonzero(mask)
            target_image = image_c3
    else:
        target_image = image_c2

    # Calculate size of overlap
    masked_image = target_image * mask
    size = xp.count_nonzero(masked_image)

    return int(size)  # Ensure we return a Python integer

def calculate_intensity_stats(intensity_image, mask):
    """
    Calculate intensity statistics for a masked region.
    
    Args:
        intensity_image: Raw intensity image
        mask: Boolean mask defining the region
    
    Returns:
        dict: Dictionary with mean, median, std, max, min intensity
    """
    xp = cp if GPU_AVAILABLE else np
    
    # Get intensity values within the mask
    intensity_values = intensity_image[mask]
    
    if len(intensity_values) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0
        }
    
    stats = {
        'mean': float(xp.mean(intensity_values)),
        'median': float(xp.median(intensity_values)),
        'std': float(xp.std(intensity_values)),
        'max': float(xp.max(intensity_values)),
        'min': float(xp.min(intensity_values))
    }
    
    # Convert from GPU to CPU if needed
    if GPU_AVAILABLE:
        stats = {k: float(v.get()) if hasattr(v, 'get') else float(v) for k, v in stats.items()}
    
    return stats

def count_positive_objects(image_c2, intensity_c3, mask_roi, threshold_method='percentile', threshold_value=75.0):
    """
    Count Channel 2 objects that are positive for Channel 3 signal.
    
    Args:
        image_c2: Label image of Channel 2 (e.g., nuclei)
        intensity_c3: Intensity image of Channel 3 (e.g., KI67)
        mask_roi: Boolean mask for the ROI from Channel 1
        threshold_method: 'percentile' or 'absolute'
        threshold_value: Threshold value (0-100 for percentile, or absolute intensity)
    
    Returns:
        dict: Dictionary with counts and threshold info
    """
    xp = cp if GPU_AVAILABLE else np
    
    # Get all unique Channel 2 objects in the ROI
    c2_in_roi = image_c2 * mask_roi
    c2_labels = xp.unique(c2_in_roi)
    c2_labels = c2_labels[c2_labels != 0]  # Remove background
    
    if len(c2_labels) == 0:
        return {
            'total_c2_objects': 0,
            'positive_c2_objects': 0,
            'negative_c2_objects': 0,
            'percent_positive': 0.0,
            'threshold_used': 0.0
        }
    
    # Calculate threshold
    if threshold_method == 'percentile':
        # Calculate threshold from all Channel 3 intensity values within ROI where Channel 2 exists
        mask_c2_in_roi = (c2_in_roi > 0)
        intensity_in_c2 = intensity_c3[mask_c2_in_roi]
        if len(intensity_in_c2) > 0:
            threshold = float(xp.percentile(intensity_in_c2, threshold_value))
        else:
            threshold = 0.0
    else:  # absolute
        threshold = threshold_value
    
    # Count positive objects
    positive_count = 0
    for label_id in c2_labels:
        # Get mask for this specific Channel 2 object
        mask_c2_obj = (image_c2 == label_id) & mask_roi
        
        # Get mean intensity of Channel 3 in this Channel 2 object
        intensity_in_obj = intensity_c3[mask_c2_obj]
        if len(intensity_in_obj) > 0:
            mean_intensity = float(xp.mean(intensity_in_obj))
            if mean_intensity >= threshold:
                positive_count += 1
    
    total_count = int(len(c2_labels))
    negative_count = total_count - positive_count
    percent_positive = (positive_count / total_count * 100) if total_count > 0 else 0.0
    
    # Convert from GPU to CPU if needed
    if GPU_AVAILABLE:
        threshold = float(threshold.get()) if hasattr(threshold, 'get') else float(threshold)
    
    return {
        'total_c2_objects': total_count,
        'positive_c2_objects': positive_count,
        'negative_c2_objects': negative_count,
        'percent_positive': percent_positive,
        'threshold_used': threshold
    }

def load_and_resize_images(file_lists, channels, image_index, no_resize):
    """Load and resize images for all channels."""
    try:
        # First load all images to get their shapes
        images = []
        shapes = []
        
        for channel in channels:
            path = file_lists[channel][image_index]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
                
            img = io.imread(path)
            images.append(img)
            shapes.append(img.shape)
        
        # Check if all shapes are the same
        shapes_equal = all(shape == shapes[0] for shape in shapes)
        
        if not shapes_equal:
            if no_resize:
                raise ValueError("Images have different shapes and --no_resize is enabled")
            
            # Resize all images to match the first one
            target_shape = shapes[0]
            print(f"Resizing images to {target_shape}")
            
            resized_images = []
            for channel in channels:
                path = file_lists[channel][image_index]
                img = load_image(path, target_shape)
                resized_images.append(img)
                
            return resized_images, True
        else:
            # Load images with GPU support if available
            loaded_images = []
            for channel in channels:
                path = file_lists[channel][image_index]
                img = load_image(path)
                loaded_images.append(img)
                
            return loaded_images, False
            
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return None, False

def process_single_roi(file_path, label_id, image_c1, image_c2, image_c3, channels, get_sizes, roi_sizes, channel3_is_labels='y', image_c3_intensity=None, count_positive='n', threshold_method='percentile', threshold_value=75.0):
    """Process a single ROI for colocalization analysis."""
    xp = cp if GPU_AVAILABLE else np
    
    # Create masks once
    mask_roi = image_c1 == label_id
    mask_c2 = image_c2 != 0
    
    # Calculate counts 
    c2_in_c1_count = count_unique_nonzero(image_c2, mask_roi & mask_c2)
    
    # Build the result row
    row = [os.path.basename(file_path), int(label_id), c2_in_c1_count]
    
    # Add size information if requested
    if get_sizes.lower() == 'y':
        size = roi_sizes.get(int(label_id), 0)
        c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id)
        row.extend([size, c2_in_c1_size])
    
    # Handle third channel if present
    if image_c3 is not None:
        if channel3_is_labels.lower() == 'y':
            # Original behavior: count objects in channel 3
            mask_c3 = image_c3 != 0
            
            # Calculate third channel statistics
            c3_in_c2_in_c1_count = count_unique_nonzero(image_c3, mask_roi & mask_c2 & mask_c3)
            c3_not_in_c2_but_in_c1_count = count_unique_nonzero(image_c3, mask_roi & ~mask_c2 & mask_c3)
            
            row.extend([c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count])
            
            # Add size information for third channel if requested
            if get_sizes.lower() == 'y':
                c3_in_c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id, mask_c2=True, image_c3=image_c3)
                c3_not_in_c2_but_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id, mask_c2=False, image_c3=image_c3)
                row.extend([c3_in_c2_in_c1_size, c3_not_in_c2_but_in_c1_size])
        else:
            # New behavior: measure intensity statistics
            # Use intensity image if provided, otherwise use image_c3
            intensity_img = image_c3_intensity if image_c3_intensity is not None else image_c3
            
            # Calculate intensity where c2 is present in c1
            mask_c2_in_c1 = mask_roi & mask_c2
            stats_c2_in_c1 = calculate_intensity_stats(intensity_img, mask_c2_in_c1)
            
            # Calculate intensity where c2 is NOT present in c1
            mask_not_c2_in_c1 = mask_roi & ~mask_c2
            stats_not_c2_in_c1 = calculate_intensity_stats(intensity_img, mask_not_c2_in_c1)
            
            # Add intensity statistics to row
            row.extend([
                stats_c2_in_c1['mean'],
                stats_c2_in_c1['median'],
                stats_c2_in_c1['std'],
                stats_c2_in_c1['max'],
                stats_not_c2_in_c1['mean'],
                stats_not_c2_in_c1['median'],
                stats_not_c2_in_c1['std'],
                stats_not_c2_in_c1['max']
            ])
            
            # Count positive Channel 2 objects if requested
            if count_positive.lower() == 'y':
                positive_counts = count_positive_objects(
                    image_c2, intensity_img, mask_roi,
                    threshold_method, threshold_value
                )
                row.extend([
                    positive_counts['positive_c2_objects'],
                    positive_counts['negative_c2_objects'],
                    positive_counts['percent_positive'],
                    positive_counts['threshold_used']
                ])
    
    return row

def load_matched_images(matched_file_dict, channels, no_resize):
    """Load images for matched files across channels."""
    try:
        # First load all images to get their shapes
        images = []
        shapes = []
        
        for channel in channels:
            path = matched_file_dict[channel]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
                
            img = io.imread(path)
            images.append(img)
            shapes.append(img.shape)
        
        # Check if all shapes are the same
        shapes_equal = all(shape == shapes[0] for shape in shapes)
        
        if not shapes_equal:
            if no_resize:
                raise ValueError("Images have different shapes and --no_resize is enabled")
            
            # Resize all images to match the first one
            target_shape = shapes[0]
            print(f"Resizing images to {target_shape}")
            
            resized_images = []
            for channel in channels:
                path = matched_file_dict[channel]
                img = load_image(path, target_shape)
                resized_images.append(img)
                
            return resized_images, True
        else:
            # Load images with GPU support if available
            loaded_images = []
            for channel in channels:
                path = matched_file_dict[channel]
                img = load_image(path)
                loaded_images.append(img)
                
            return loaded_images, False
            
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return None, False

def process_batch_matched(batch_matched_files, channels, get_sizes, no_resize, batch_num, total_batches, channel3_is_labels='y', count_positive='n', threshold_method='percentile', threshold_value=75.0):
    """Process a batch of matched image file sets for colocalization analysis."""
    batch_rows = []
    xp = cp if GPU_AVAILABLE else np
    
    for matched_dict in tqdm(batch_matched_files, desc=f"Processing batch {batch_num}/{total_batches}"):
        try:
            # Get the primary file path (from first channel) for naming
            file_path = matched_dict[channels[0]]
            
            # Load and prepare images
            images, resized = load_matched_images(matched_dict, channels, no_resize)
            
            if images is None:
                print(f"Failed to load images for {os.path.basename(file_path)}")
                continue
                
            image_c1, image_c2 = images[:2]
            image_c3 = images[2] if len(channels) == 3 else None
            
            # Load intensity image for channel 3 if it's not labels
            image_c3_intensity = None
            if image_c3 is not None and channel3_is_labels.lower() == 'n':
                # The image we loaded is already the intensity image
                image_c3_intensity = image_c3
                # No separate label image for channel 3
            
            # Get unique label IDs in image_c1
            label_ids = get_nonzero_labels(image_c1)
            
            # Pre-calculate sizes for image_c1 if needed
            roi_sizes = {}
            if get_sizes.lower() == 'y':
                roi_sizes = calculate_all_rois_size(image_c1)
            
            # Process each label
            for label_id in label_ids:
                row = process_single_roi(
                    file_path, label_id, image_c1, image_c2, image_c3,
                    channels, get_sizes, roi_sizes, channel3_is_labels, image_c3_intensity,
                    count_positive, threshold_method, threshold_value
                )
                batch_rows.append(row)
                
            # Clean up GPU memory after each file
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
        except Exception as e:
            print(f"Error processing matched files: {str(e)}")
            traceback.print_exc()
            
    return batch_rows

def process_batch(batch_files, batch_start, file_lists, channels, get_sizes, no_resize, batch_num, total_batches, channel3_is_labels='y', count_positive='n', threshold_method='percentile', threshold_value=75.0):
    """Process a batch of image files for colocalization analysis (deprecated - kept for compatibility)."""
    batch_rows = []
    xp = cp if GPU_AVAILABLE else np
    
    for i, file_path in enumerate(tqdm(batch_files, desc=f"Processing batch {batch_num}/{total_batches}")):
        try:
            # Load and prepare images
            image_index = batch_start + i
            images, resized = load_and_resize_images(file_lists, channels, image_index, no_resize)
            
            if images is None:
                print(f"Failed to load images for {file_path}")
                continue
                
            image_c1, image_c2 = images[:2]
            image_c3 = images[2] if len(channels) == 3 else None
            
            # Load intensity image for channel 3 if it's not labels
            image_c3_intensity = None
            if image_c3 is not None and channel3_is_labels.lower() == 'n':
                # The image we loaded is already the intensity image
                image_c3_intensity = image_c3
                # No separate label image for channel 3
            
            # Get unique label IDs in image_c1
            label_ids = get_nonzero_labels(image_c1)
            
            # Pre-calculate sizes for image_c1 if needed
            roi_sizes = {}
            if get_sizes.lower() == 'y':
                roi_sizes = calculate_all_rois_size(image_c1)
            
            # Process each label
            for label_id in label_ids:
                row = process_single_roi(
                    file_path, label_id, image_c1, image_c2, image_c3,
                    channels, get_sizes, roi_sizes, channel3_is_labels, image_c3_intensity,
                    count_positive, threshold_method, threshold_value
                )
                batch_rows.append(row)
                
            # Clean up GPU memory after each file
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            traceback.print_exc()
            
    return batch_rows

def match_files_across_channels(file_lists, channels):
    """
    Match files across channels based on common substrings.
    
    Returns:
        list: List of dictionaries, each containing matched file paths for all channels
    """
    # Get basenames for the first channel
    first_channel = channels[0]
    matched_files = []
    
    for file_c1 in file_lists[first_channel]:
        basename_c1 = os.path.basename(file_c1)
        
        # Try to find matching files in other channels
        match_dict = {first_channel: file_c1}
        matched = True
        
        for channel in channels[1:]:
            best_match = None
            best_match_len = 0
            
            # Find the best matching file based on common substring
            for file_cn in file_lists[channel]:
                basename_cn = os.path.basename(file_cn)
                common = longest_common_substring(basename_c1, basename_cn)
                
                # Require a minimum common substring length
                if len(common) > best_match_len and len(common) >= 10:
                    best_match = file_cn
                    best_match_len = len(common)
            
            if best_match is None:
                print(f"Warning: No match found for {basename_c1} in channel {channel}")
                matched = False
                break
            
            match_dict[channel] = best_match
        
        if matched:
            matched_files.append(match_dict)
    
    return matched_files

def coloc_channels(file_lists, channels, get_sizes, size_method=None, num_workers=1, batch_size=10, no_resize=False, channel3_is_labels='y', count_positive='n', threshold_method='percentile', threshold_value=75.0):
    """
    Calculate colocalization between channels for multiple images.
    
    Returns:
        list: List of CSV rows
    """
    csv_rows = []
    
    # Match files across channels by common substring
    print("Matching files across channels...")
    matched_files = match_files_across_channels(file_lists, channels)
    print(f"Successfully matched {len(matched_files)} file sets")
    
    total_batches = (len(matched_files) - 1) // batch_size + 1
    
    # Process in batches
    for batch_idx, batch_start in enumerate(range(0, len(matched_files), batch_size)):
        batch_end = min(batch_start + batch_size, len(matched_files))
        batch_matched_files = matched_files[batch_start:batch_end]
        
        # Process each matched file set in the batch
        batch_rows = process_batch_matched(
            batch_matched_files, channels, 
            get_sizes, no_resize, batch_idx + 1, total_batches, channel3_is_labels,
            count_positive, threshold_method, threshold_value
        )
        
        csv_rows.extend(batch_rows)
        gc.collect()  # Force garbage collection between batches
        
    return csv_rows

def main():
    try:
        args = parse_args()
        parent_dir = args.input
        channels = args.channels
        label_patterns = args.label_patterns
        get_sizes = args.get_sizes
        size_method = args.size_method if get_sizes.lower() == 'y' else None
        num_workers = args.num_workers
        batch_size = args.batch_size
        no_resize = args.no_resize
        channel3_is_labels = args.channel3_is_labels if len(channels) == 3 else 'y'
        count_positive = args.count_positive if len(channels) == 3 and channel3_is_labels.lower() == 'n' else 'n'
        threshold_method = args.threshold_method
        threshold_value = args.threshold_value

        print(f"Configuration: channels={channels}, get_sizes={get_sizes}, "
              f"size_method={size_method}, num_workers={num_workers}, batch_size={batch_size}, no_resize={no_resize}, "
              f"channel3_is_labels={channel3_is_labels if len(channels) == 3 else 'N/A'}")
        
        if count_positive.lower() == 'y':
            print(f"Positive counting enabled: {threshold_method} threshold = {threshold_value}")

        # Validate channel inputs
        if len(set(channels)) < len(channels):
            raise ValueError("Channel names must be unique")
            
        if len(channels) < MIN_CHANNELS or len(channels) > MAX_CHANNELS:
            raise ValueError(f"Between {MIN_CHANNELS} and {MAX_CHANNELS} channels must be provided")

        # Validate label patterns
        if len(label_patterns) != len(channels):
            raise ValueError("Number of label patterns must match number of channels")

        # Get file lists for each channel
        file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern)))
                     for channel, label_pattern in zip(channels, label_patterns)}

        # Validate files before processing
        validate_file_lists(file_lists, channels)

        print(f"Found {len(file_lists[channels[0]])} files for processing")

        # Use context manager for memory operations
        with memory_manager():
            csv_rows = coloc_channels(file_lists, channels, get_sizes, size_method, num_workers, batch_size, no_resize, channel3_is_labels, count_positive, threshold_method, threshold_value)

        # Create the filename using the selected channels
        channel_string = '_'.join(channels)  # Concatenate channel names with underscores
        csv_file = os.path.join(parent_dir, f'{channel_string}_colocalization.csv')

        # Define header based on the arguments
        header = ['Filename', f'{channels[0]}_label_id', f'{channels[1]}_in_{channels[0]}_count']
        if get_sizes.lower() == 'y':
            header.extend([f'{channels[0]}_size', f'{channels[1]}_in_{channels[0]}_size'])
        if len(channels) == 3:
            if channel3_is_labels.lower() == 'y':
                # Label-based mode for channel 3
                header.extend([f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_count',
                               f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_count'])
                if get_sizes.lower() == 'y':
                    header.extend([f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_size',
                                   f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_size'])
            else:
                # Intensity-based mode for channel 3
                header.extend([
                    f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_mean',
                    f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_median',
                    f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_std',
                    f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_max',
                    f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_mean',
                    f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_median',
                    f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_std',
                    f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_max'
                ])
                
                # Add positive counting columns if requested
                if count_positive.lower() == 'y':
                    header.extend([
                        f'{channels[1]}_in_{channels[0]}_positive_for_{channels[2]}_count',
                        f'{channels[1]}_in_{channels[0]}_negative_for_{channels[2]}_count',
                        f'{channels[1]}_in_{channels[0]}_percent_positive_for_{channels[2]}',
                        f'{channels[2]}_threshold_used'
                    ])

        # Write to CSV
        with open(csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            csv_writer.writerows(csv_rows)

        print(f'Colocalization analysis complete. Results saved to {csv_file}')

    except ValueError as e:
        print(f"ValueError: {e}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()