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
                         help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    
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
        if GPU_AVAILABLE and 0 in unique_vals.get() or not GPU_AVAILABLE and 0 in unique_vals:
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

def process_single_roi(file_path, label_id, image_c1, image_c2, image_c3, channels, get_sizes, roi_sizes):
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
    
    return row

def process_batch(batch_files, batch_start, file_lists, channels, get_sizes, no_resize, batch_num, total_batches):
    """Process a batch of image files for colocalization analysis."""
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
                    channels, get_sizes, roi_sizes
                )
                batch_rows.append(row)
                
            # Clean up GPU memory after each file
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            traceback.print_exc()
            
    return batch_rows

def coloc_channels(file_lists, channels, get_sizes, size_method=None, num_workers=1, batch_size=10, no_resize=False):
    """
    Calculate colocalization between channels for multiple images.
    
    Returns:
        list: List of CSV rows
    """
    csv_rows = []
    file_paths = file_lists[channels[0]]
    total_batches = (len(file_paths) - 1) // batch_size + 1
    
    # Process in batches
    for batch_idx, batch_start in enumerate(range(0, len(file_paths), batch_size)):
        batch_end = min(batch_start + batch_size, len(file_paths))
        batch_files = file_paths[batch_start:batch_end]
        
        # Process each file in the batch
        batch_rows = process_batch(
            batch_files, batch_start, file_lists, channels, 
            get_sizes, no_resize, batch_idx + 1, total_batches
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

        print(f"Configuration: channels={channels}, get_sizes={get_sizes}, "
              f"size_method={size_method}, num_workers={num_workers}, batch_size={batch_size}, no_resize={no_resize}")

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
            csv_rows = coloc_channels(file_lists, channels, get_sizes, size_method, num_workers, batch_size, no_resize)

        # Create the filename using the selected channels
        channel_string = '_'.join(channels)  # Concatenate channel names with underscores
        csv_file = os.path.join(parent_dir, f'{channel_string}_colocalization.csv')

        # Define header based on the arguments
        header = ['Filename', f'{channels[0]}_label_id', f'{channels[1]}_in_{channels[0]}_count']
        if get_sizes.lower() == 'y':
            header.extend([f'{channels[0]}_size', f'{channels[1]}_in_{channels[0]}_size'])
        if len(channels) == 3:
            header.extend([f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_count',
                           f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_count'])
            if get_sizes.lower() == 'y':
                header.extend([f'{channels[2]}_in_{channels[1]}_in_{channels[0]}_size',
                               f'{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}_size'])

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