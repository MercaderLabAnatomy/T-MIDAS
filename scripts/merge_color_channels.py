import os
import argparse
import glob
from tqdm import tqdm
from tifffile import imwrite, TiffFile
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Batch merge channels with simple sorting')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--time_steps', type=int, default=None, help='Number of time steps if time-lapse (leave empty if static)')
    parser.add_argument('--is_3d', action='store_true', help='Images are 3D (Z dimension present)')
    return parser.parse_args()

def natural_sort_key(text):
    """Generate a key for natural sorting (handles numbers correctly)"""
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', text)]

def infer_dimension_order(shape, time_steps, is_3d):
    """Infer dimension order from shape, ensuring channel comes first after time"""
    is_timelapse = time_steps is not None
    
    if is_timelapse:
        if is_3d:
            return 'TCZYX'  # Time, Channel, Z, Y, X
        else:
            return 'TCYX'   # Time, Channel, Y, X
    else:
        if is_3d:
            return 'CZYX'   # Channel, Z, Y, X
        else:
            return 'CYX'    # Channel, Y, X

def merge_channels_cpu(file_lists, channels, time_steps, is_3d, merged_dir):
    num_channels = len(channels)
    is_timelapse = time_steps is not None
    
    # Check that all channels have the same number of files
    file_counts = [len(file_lists[channel]) for channel in channels]
    if len(set(file_counts)) > 1:
        raise ValueError(f"Channels have different numbers of files: {dict(zip(channels, file_counts))}")
    
    num_files = file_counts[0]
    if num_files == 0:
        raise ValueError("No files found in any channel")
    
    # Get information about the first image to determine structure
    with TiffFile(file_lists[channels[0]][0]) as tif:
        img = tif.asarray()

    print(f"Input image shape: {img.shape}")
    print(f"Structure: {'Time-lapse' if is_timelapse else 'Static'}{f' ({time_steps} steps)' if is_timelapse else ''}, {'3D' if is_3d else '2D'}")

    # Validate time dimension if specified
    if is_timelapse and img.shape[0] != time_steps:
        print(f"Warning: Expected {time_steps} time steps but found {img.shape[0]} in first dimension")

    # Determine output shape - channel comes first (after time if present)
    if is_timelapse:
        if is_3d:
            # Input: T,Z,Y,X -> Output: T,C,Z,Y,X
            time_points, z_slices, height, width = img.shape
            merged_shape = (time_points, num_channels, z_slices, height, width)
        else:
            # Input: T,Y,X -> Output: T,C,Y,X
            time_points, height, width = img.shape
            merged_shape = (time_points, num_channels, height, width)
    else:
        if is_3d:
            # Input: Z,Y,X -> Output: C,Z,Y,X
            z_slices, height, width = img.shape
            merged_shape = (num_channels, z_slices, height, width)
        else:
            # Input: Y,X -> Output: C,Y,X
            height, width = img.shape
            merged_shape = (num_channels, height, width)

    output_dim_order = infer_dimension_order(merged_shape, time_steps, is_3d)
    print(f"Output shape: {merged_shape}, dimension order: {output_dim_order}")

    # Process each file index
    for i in tqdm(range(num_files), desc='Merging files'):
        merged_img = np.zeros(merged_shape, dtype=img.dtype)
        
        # Load corresponding file from each channel
        for c, channel in enumerate(channels):
            with TiffFile(file_lists[channel][i]) as tif:
                channel_img = tif.asarray()
                
                # Place each channel in the correct position
                if is_timelapse:
                    if is_3d:
                        merged_img[:, c, :, :, :] = channel_img
                    else:
                        merged_img[:, c, :, :] = channel_img
                else:
                    if is_3d:
                        merged_img[c, :, :, :] = channel_img
                    else:
                        merged_img[c, :, :] = channel_img

        # Generate output filename based on the first channel's filename
        base_filename = os.path.basename(file_lists[channels[0]][i])
        # Remove channel name from filename if present
        clean_filename = base_filename
        for channel in channels:
            clean_filename = clean_filename.replace(channel, "").replace("--", "-").strip("-")
        
        # Ensure we have a reasonable filename
        if not clean_filename or clean_filename == ".tif":
            clean_filename = f"merged_{i:03d}.tif"
        
        output_filename = os.path.join(merged_dir, clean_filename)
        
        # Save with proper metadata indicating channel-first format
        imwrite(output_filename, merged_img, compression='zlib', imagej=True, 
                metadata={'axes': output_dim_order})
    
    print(f"Merging completed successfully. {num_files} files saved.")

def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    time_steps = args.time_steps
    is_3d = args.is_3d
    is_timelapse = time_steps is not None

    # Get sorted list of files for each channel (natural sort for proper numeric ordering)
    file_lists = {}
    for channel in channels:
        pattern = os.path.join(parent_dir, channel, '*.tif')
        all_files = sorted(glob.glob(pattern), key=natural_sort_key)
        # Filter out label files
        file_lists[channel] = [f for f in all_files if not f.endswith('_labels.tif')]

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    print("Configuration:")
    print(f"  Time-lapse: {time_steps if is_timelapse else 'No'}{f' steps' if is_timelapse else ''}")
    print(f"  3D: {is_3d}")
    print(f"  Expected input structure: {'T' if is_timelapse else ''}{'Z' if is_3d else ''}YX")
    print(f"  Output structure: {'T' if is_timelapse else ''}C{'Z' if is_3d else ''}YX")
    print("\nFiles found per channel:")
    for channel in channels:
        print(f"  {channel}: {len(file_lists[channel])}")
        if file_lists[channel]:
            print(f"    First: {os.path.basename(file_lists[channel][0])}")
            print(f"    Last:  {os.path.basename(file_lists[channel][-1])}")

    # Create output directory
    merged_dir = os.path.join(parent_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    merge_channels_cpu(file_lists, channels, time_steps, is_3d, merged_dir)
    print(f"Merged images saved in {merged_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)