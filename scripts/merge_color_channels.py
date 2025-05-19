import os
import argparse
import glob
from tqdm import tqdm
from tifffile import imwrite, TiffFile
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Batch merge channels')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--time_steps', type=int, default=None, nargs='?', help='Number of time steps for timelapse images. Leave empty if not a timelapse.')
    return parser.parse_args()

def find_channel_axis(shape):
    """Find channel axis - should be first or last dim with size <= 4"""
    # Check first dimension
    if shape[0] <= 4:
        return 0
    
    # Check last dimension
    if shape[-1] <= 4:
        return len(shape) - 1
    
    # If neither first nor last, check all dimensions for compatibility
    for i, dim_size in enumerate(shape):
        if dim_size <= 4:
            return i
    
    return None

def infer_dimension_order(shape, num_channels, time_steps):
    """Infer dimension order from shape, ensuring channel comes first"""
    dim_order = ''
    
    if len(shape) == 4:
        if time_steps and shape[0] == time_steps:
            dim_order = 'TCYX'  # Time, Channel, Y, X
        else:
            dim_order = 'CZYX'  # Channel, Z, Y, X
    elif len(shape) == 3:
        if time_steps and shape[0] == time_steps:
            dim_order = 'TCY'   # Time, Channel, Y (treating as 2D with time)
        else:
            dim_order = 'CYX'   # Channel, Y, X
    elif len(shape) == 2:
        dim_order = 'YX'     # Y, X (single channel)
    elif len(shape) == 5:
        dim_order = 'TCZYX'  # Time, Channel, Z, Y, X
    else:
        raise ValueError(f"Unsupported image shape: {shape}")
    
    return dim_order

def merge_channels_cpu(file_lists, channels, time_steps, merged_dir):
    num_channels = len(channels)
    
    # Get information about the first image
    with TiffFile(file_lists[channels[0]][0]) as tif:
        img = tif.asarray()
        metadata = tif.imagej_metadata or {}

    # Try to get dimension order from metadata
    input_dim_order = metadata.get('axes', '') if metadata else ''
    
    # Infer the structure of the input images
    is_3d = len(img.shape) >= 3 and (img.shape[-3] > 4 if len(img.shape) >= 3 else False)
    is_time_series = time_steps is not None
    
    print(f"Input image shape: {img.shape}, input dimension order: {input_dim_order}")
    print(f"Detected: {'3D' if is_3d else '2D'}, {'time series' if is_time_series else 'static'}")

    # Determine output shape - always put channel first
    if is_time_series:
        if is_3d:
            # Input: T,Z,Y,X -> Output: T,C,Z,Y,X
            time_points, z_slices, height, width = img.shape
            merged_shape = (time_points, num_channels, z_slices, height, width)
            output_dim_order = 'TCZYX'
        else:
            # Input: T,Y,X -> Output: T,C,Y,X
            time_points, height, width = img.shape
            merged_shape = (time_points, num_channels, height, width)
            output_dim_order = 'TCYX'
    else:
        if is_3d:
            # Input: Z,Y,X -> Output: C,Z,Y,X
            z_slices, height, width = img.shape
            merged_shape = (num_channels, z_slices, height, width)
            output_dim_order = 'CZYX'
        else:
            # Input: Y,X -> Output: C,Y,X
            height, width = img.shape
            merged_shape = (num_channels, height, width)
            output_dim_order = 'CYX'

    print(f"Output shape: {merged_shape}, output dimension order: {output_dim_order}")

    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging files'):
        merged_img = np.zeros(merged_shape, dtype=img.dtype)
        
        for c, channel in enumerate(channels):
            with TiffFile(file_lists[channel][i]) as tif:
                channel_img = tif.asarray()
                
                # Place each channel in the correct position (channel axis is always second from the left after time)
                if is_time_series:
                    if is_3d:
                        merged_img[:, c, :, :, :] = channel_img
                    else:
                        merged_img[:, c, :, :] = channel_img
                else:
                    if is_3d:
                        merged_img[c, :, :, :] = channel_img
                    else:
                        merged_img[c, :, :] = channel_img

        # Generate output filename
        base_filename = os.path.basename(file_lists[channels[0]][i])
        # Remove the channel name from the filename if it exists
        for channel in channels:
            base_filename = base_filename.replace(f"{channel}-", "").replace(f"-{channel}", "").replace(channel, "")
        # Clean up any double dashes or leading/trailing dashes
        base_filename = base_filename.replace("--", "-").strip("-")
        
        output_filename = os.path.join(merged_dir, base_filename)
        print(f"Saving merged image to {output_filename}")
        
        # Save with proper metadata indicating channel-first format
        imwrite(output_filename, merged_img, compression='zlib', imagej=True, 
                metadata={'axes': output_dim_order})
    
    print("Merging completed successfully.")

def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    time_steps = args.time_steps

    # Get a list of files for each channel
    file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, '*.tif'))) for channel in channels}
    file_lists = {channel: [f for f in file_lists[channel] if not f.endswith('_labels.tif')] for channel in file_lists}

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    print("Number of images in each channel:")
    for channel in file_lists:
        print(f"{channel}: {len(file_lists[channel])}")

    # Check if all channels have the same number of files
    if len(set(len(file_lists[channel]) for channel in channels)) > 1:
        raise ValueError("All channels must have the same number of files.")

    # Create a new folder to save the merged images
    merged_dir = os.path.join(parent_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    merge_channels_cpu(file_lists, channels, time_steps, merged_dir)

    print("Merged images saved in", merged_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)