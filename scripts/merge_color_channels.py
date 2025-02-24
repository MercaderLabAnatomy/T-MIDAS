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

def infer_dimension_order(shape, num_channels, time_steps):
    dim_order = ''
    if len(shape) == 4:
        if time_steps and shape[0] == time_steps:
            dim_order = 'TZYX'
        else:
            dim_order = 'ZCYX'
    elif len(shape) == 3:
        if time_steps and shape[0] == time_steps:
            dim_order = 'TYX'
        else:
            dim_order = 'ZYX'
    elif len(shape) == 2:
        dim_order = 'YX'
    else:
        raise ValueError(f"Unsupported image shape: {shape}")
    
    return dim_order

def merge_channels_cpu(file_lists, channels, time_steps, merged_dir):
    num_channels = len(channels)
    
    # Get information about the first image
    with TiffFile(file_lists[channels[0]][0]) as tif:
        img = tif.asarray()
        metadata = tif.imagej_metadata

    # Try to get dimension order from metadata
    dim_order = metadata.get('axes', '') if metadata else ''
    
    if not dim_order:
        # Infer dimension order from shape and user info
        dim_order = infer_dimension_order(img.shape, num_channels, time_steps)

    print(f"Input image shape: {img.shape}, detected dimension order: {dim_order}")

    is_3d = 'Z' in dim_order
    is_time_series = 'T' in dim_order

    if is_time_series:
        if is_3d:
            time_points, z_slices, height, width = img.shape
            merged_shape = (time_points, num_channels, z_slices, height, width)
        else:
            time_points, height, width = img.shape
            merged_shape = (time_points, num_channels, height, width)
    else:
        if is_3d:
            z_slices, height, width = img.shape
            merged_shape = (num_channels, z_slices, height, width)
        else:
            height, width = img.shape
            merged_shape = (num_channels, height, width)

    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging files'):
        merged_img = np.zeros(merged_shape, dtype=img.dtype)
        
        for c, channel in enumerate(channels):
            with TiffFile(file_lists[channel][i]) as tif:
                channel_img = tif.asarray()
                
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

        # Reorder dimensions
        if is_time_series:
            if is_3d:
                merged_img = np.moveaxis(merged_img, 1, 2)  # TCZYX
            else:
                merged_img = np.moveaxis(merged_img, 1, 1)  # TCYX
        else:
            if is_3d:
                merged_img = np.moveaxis(merged_img, 0, 1)  # ZCYX
            else:
                merged_img = np.moveaxis(merged_img, 0, 0)  # CYX
        
        output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0], '').lstrip('-'))
        print(f"Saving merged image to {output_filename}")
        output_dim_order = 'TZCYX' if is_time_series and is_3d else 'TCYX' if is_time_series else 'ZCYX' if is_3d else 'CYX'
        imwrite(output_filename, merged_img, compression='zlib', imagej=True, metadata={'axes': output_dim_order})
    
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
