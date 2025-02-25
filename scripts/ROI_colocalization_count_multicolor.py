import os
import glob
import csv
import argparse
import traceback
import gc
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor

try:
    import cupy as cp
    import cucim.skimage as cusk
    from skimage import io, measure

    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    from skimage import io, measure

    GPU_AVAILABLE = False


@contextmanager
def gpu_memory_manager():
    """Context manager to handle GPU memory cleanup after operations."""
    try:
        yield
    finally:
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True,
                        help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    parser.add_argument('--get_sizes', type=str, default='n', help='Do you want to get sizes of ROIs in all channels? (y/n)')
    parser.add_argument('--size_method', type=str, choices=['median', 'sum'],
                        help='Method to calculate sizes for second and third channels: "median" or "sum" (only used if get_sizes is "y")')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker processes to use when GPU is not available')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of images to process in each batch')

    args = parser.parse_args()

    if args.get_sizes.lower() == 'y' and args.size_method is None:
        args.size_method = input("\nWhich size stats? Type median or sum: ")
        while args.size_method not in ['median', 'sum']:
            print("Invalid input. Please enter 'median' or 'sum'.")
            args.size_method = input("\nWhich size stats? Type median or sum: ")

    return args

def validate_file_lists(file_lists, channels):
    """Ensures all channels have the same number of files and they exist."""
    lengths = [len(file_lists[channel]) for channel in channels]
    if len(set(lengths)) != 1:
        raise ValueError(f"Channel folders contain different numbers of files: {lengths}")
    
    # Check file existence
    for channel in channels:
        for filepath in file_lists[channel]:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
    
    return True


def load_image(file_path):
    """Load an image file with proper error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        image = io.imread(file_path)
        if image is None or image.size == 0:
            raise ValueError(f"Failed to load image or empty image: {file_path}")
            
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


def coloc_counts(image_c1, image_c2, label_id, image_c3=None):
    """
    Calculate colocalization counts for a specific ROI label.
    
    Args:
        image_c1: First channel image with ROI labels
        image_c2: Second channel image
        label_id: Label ID to analyze
        image_c3: Optional third channel image
        
    Returns:
        dict: Dictionary containing colocalization results
    """
    xp = cp if GPU_AVAILABLE else np
    
    # Create mask for the specific ROI
    mask_roi = image_c1 == label_id
    
    # Create boolean masks for nonzero entries within this ROI
    mask_c2 = image_c2 != 0
    
    # Count unique c2 labels within this specific c1 ROI
    c2_in_c1_count = len(xp.unique(image_c2[mask_roi & mask_c2]))
    if c2_in_c1_count > 0 and 0 in xp.unique(image_c2[mask_roi & mask_c2]):
        c2_in_c1_count -= 1  # Subtract 1 if 0 is included in the unique values
    
    results = {
        "c2_in_c1_count": c2_in_c1_count
    }

    # If a third channel is provided, calculate additional stats
    if image_c3 is not None:
        mask_c3 = image_c3 != 0
        
        # Calculate counts specifically for this ROI
        c3_in_c2_in_c1_count = len(xp.unique(image_c3[mask_roi & mask_c2 & mask_c3]))
        if c3_in_c2_in_c1_count > 0 and 0 in xp.unique(image_c3[mask_roi & mask_c2 & mask_c3]):
            c3_in_c2_in_c1_count -= 1
            
        c3_not_in_c2_but_in_c1_count = len(xp.unique(image_c3[mask_roi & ~mask_c2 & mask_c3]))
        if c3_not_in_c2_but_in_c1_count > 0 and 0 in xp.unique(image_c3[mask_roi & ~mask_c2 & mask_c3]):
            c3_not_in_c2_but_in_c1_count -= 1

        results["c3_in_c2_in_c1_count"] = c3_in_c2_in_c1_count
        results["c3_not_in_c2_but_in_c1_count"] = c3_not_in_c2_but_in_c1_count

    return results


      
      
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

def coloc_channels(file_lists, channels, get_sizes, size_method=None, num_workers=1, batch_size=10):
    """
    Calculate colocalization between channels for multiple images.
    
    Args:
        file_lists: Dictionary mapping channel names to lists of file paths
        channels: List of channel names
        get_sizes: Whether to calculate sizes
        size_method: Method to calculate sizes
        num_workers: Number of worker processes to use
        batch_size: Number of images to process in each batch
        
    Returns:
        list: List of CSV rows
    """
    csv_rows = []
    file_paths = file_lists[channels[0]]
    xp = cp if GPU_AVAILABLE else np
    
    # Process in batches to manage memory better
    for batch_start in range(0, len(file_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(file_paths))
        batch_files = file_paths[batch_start:batch_end]
        batch_rows = []
        
        for i, file_path in enumerate(tqdm(batch_files, desc=f"Processing batch {batch_start//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1}")):
            try:
                # Check file sizes before loading to avoid memory issues
                file_size = os.path.getsize(file_path)
                if file_size > 1e9:  # 1 GB threshold
                    print(f"Warning: Large file detected ({file_size/1e6:.1f} MB): {file_path}")
                
                # Load images with better error handling
                image_index = batch_start + i
                images = []
                for channel in channels:
                    try:
                        channel_path = file_lists[channel][image_index]
                        img = load_image(channel_path)
                        images.append(img)
                    except Exception as e:
                        raise ValueError(f"Error loading {channel} image: {str(e)}")
                
                image_c1, image_c2 = images[:2]
                image_c3 = images[2] if len(channels) == 3 else None

                # Find unique label IDs in image_c1
                mask_c1 = image_c1 != 0
                label_ids = xp.unique(image_c1[mask_c1])
                label_ids = label_ids[label_ids != 0]  # remove zero label
                
                if GPU_AVAILABLE:
                    label_ids = [int(x) for x in label_ids.get()]  # Convert to Python integers if using GPU
                else:
                    label_ids = [int(x) for x in label_ids]  # Convert to Python integers

                # Pre-calculate sizes for each label in image_c1 only once
                image_c1_sizes = {}  # Initialize to prevent errors
                if get_sizes.lower() == 'y':
                    image_c1_sizes = calculate_all_rois_size(image_c1)

                for label_id in label_ids:
                    label_id_int = int(label_id)  # Convert to Python integer
                    
                    # Create mask for this specific ROI
                    mask_roi = image_c1 == label_id_int
                    mask_c2 = image_c2 != 0
                    
                    # Calculate counts specifically for this ROI
                    c2_in_c1_count = len(xp.unique(image_c2[mask_roi & mask_c2]))
                    # Remove 0 from count if present
                    if c2_in_c1_count > 0:
                        unique_vals = xp.unique(image_c2[mask_roi & mask_c2])
                        if GPU_AVAILABLE and 0 in unique_vals.get() or not GPU_AVAILABLE and 0 in unique_vals:
                            c2_in_c1_count -= 1
                    
                    row = [os.path.basename(file_path), label_id_int, c2_in_c1_count]

                    if get_sizes.lower() == 'y':
                        size = image_c1_sizes.get(label_id_int, 0)
                        c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id_int)
                        row.extend([size, c2_in_c1_size])

                    if image_c3 is not None:
                        mask_c3 = image_c3 != 0
                        
                        # Calculate counts for third channel within this specific ROI
                        c3_in_c2_in_c1_count = len(xp.unique(image_c3[mask_roi & mask_c2 & mask_c3]))
                        # Remove 0 from count if present
                        if c3_in_c2_in_c1_count > 0:
                            unique_vals = xp.unique(image_c3[mask_roi & mask_c2 & mask_c3])
                            if GPU_AVAILABLE and 0 in unique_vals.get() or not GPU_AVAILABLE and 0 in unique_vals:
                                c3_in_c2_in_c1_count -= 1
                                
                        c3_not_in_c2_but_in_c1_count = len(xp.unique(image_c3[mask_roi & ~mask_c2 & mask_c3]))
                        # Remove 0 from count if present
                        if c3_not_in_c2_but_in_c1_count > 0:
                            unique_vals = xp.unique(image_c3[mask_roi & ~mask_c2 & mask_c3])
                            if GPU_AVAILABLE and 0 in unique_vals.get() or not GPU_AVAILABLE and 0 in unique_vals:
                                c3_not_in_c2_but_in_c1_count -= 1
                        
                        row.extend([c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count])
                        
                        if get_sizes.lower() == 'y':
                            c3_in_c2_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id_int, mask_c2=True, image_c3=image_c3)
                            c3_not_in_c2_but_in_c1_size = calculate_coloc_size(image_c1, image_c2, label_id_int, mask_c2=False, image_c3=image_c3)
                            row.extend([c3_in_c2_in_c1_size, c3_not_in_c2_but_in_c1_size])
                    
                    batch_rows.append(row)

                # Cleanup GPU memory after each file if needed
                if GPU_AVAILABLE:
                    gc.collect()
                    cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                traceback.print_exc()
        
        # Append batch results
        csv_rows.extend(batch_rows)
        
        # Force garbage collection between batches
        gc.collect()
    
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

        print(f"Configuration: channels={channels}, get_sizes={get_sizes}, " 
              f"size_method={size_method}, num_workers={num_workers}, batch_size={batch_size}")

        if len(set(channels)) < len(channels) or len(channels) < 2 or len(channels) > 3:
            raise ValueError("Channel names must be unique and 2 or 3 channels must be provided.")

        file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) 
                     for channel, label_pattern in zip(channels, label_patterns)}
        
        # Validate files before processing
        validate_file_lists(file_lists, channels)
        
        print(f"Found {len(file_lists[channels[0]])} files for processing")
        
        # Use context manager for GPU operations
        with gpu_memory_manager():
            csv_rows = coloc_channels(file_lists, channels, get_sizes, size_method, num_workers, batch_size)

        # Create the filename using the selected channels
        channel_string = '_'.join(channels)  # Concatenate channel names with underscores
        csv_file = os.path.join(parent_dir, f'{channel_string}_colocalization.csv')  # create the filename with channel names

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Filename', f"{channels[0]}_ROI_ID"]
            header.extend([f"{channels[1]}_in_{channels[0]}_count"])
            if get_sizes.lower() == 'y':
                header.extend([f"{channels[0]}_size"])
                if size_method == 'sum':
                    header.extend([f"{channels[1]}_in_{channels[0]}_total_size"])
                if size_method == 'median':
                    header.extend([f"{channels[1]}_in_{channels[0]}_median_size"])
            if len(channels) == 3:
                header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_count",
                              f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_count"])
                if get_sizes.lower() == 'y':
                    if size_method == 'sum':
                        header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_total_size",
                                      f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_total_size"])
                    if size_method == 'median':
                        header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_median_size",
                                      f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_median_size"])

            writer.writerow(header)
            writer.writerows(csv_rows)

        print(f"Colocalization results saved to {csv_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    if GPU_AVAILABLE:
        print("GPU acceleration is available and will be used.")
    else:
        print("GPU acceleration is not available. Using CPU.")
    main()