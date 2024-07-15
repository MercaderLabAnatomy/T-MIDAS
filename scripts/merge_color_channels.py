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
    parser.add_argument('--dim_order', type=str, default='YX', help='Dimension order of the input images.')
    return parser.parse_args()

def merge_channels_cpu(file_lists, channels, dim_order, merged_dir):
    # Get information about the first image
    with TiffFile(file_lists[channels[0]][0]) as tif:
        series = tif.series[0]
        img_shape = series.shape
        dtype = series.dtype
    print(f"Input image shape: {img_shape}, dimension order: {dim_order}, dtype: {dtype}")
    
    if dim_order not in ['YX', 'ZYX']:
        raise ValueError(f"Expected dimension order 'YX' or 'ZYX', but got '{dim_order}'")
    
    is_3d = len(img_shape) == 3 and dim_order == 'ZYX'
    
    if is_3d:
        z_slices, height, width = img_shape
        merged_shape = (len(channels), z_slices, height, width)
    else:
        height, width = img_shape
        merged_shape = (len(channels), height, width)
    
    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging files'):
        merged_img = np.zeros(merged_shape, dtype=dtype)
        
        for c, channel in enumerate(channels):
            print(f"Processing channel: {channel}")
            with TiffFile(file_lists[channel][i]) as tif:
                if is_3d:
                    merged_img[c,:,:,:] = tif.asarray()
                else:
                    merged_img[c,:,:] = tif.asarray()

        # reorder 2D to CYX and 3D to ZCYX
        if is_3d:
            merged_img = np.moveaxis(merged_img, 0, 1)
        else:
            merged_img = np.moveaxis(merged_img, 0, 0)  # No need to move axis for 2D
        
        output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0], ''))
        imwrite(output_filename, merged_img, compression='zlib', imagej=True, metadata={'axes': 'ZCYX' if is_3d else 'CYX'})
    
    print("Merging completed successfully.")

def merge_channels_time_series_cpu(file_lists, channels, dim_order, merged_dir):
    # Get information about the first image
    with TiffFile(file_lists[channels[0]][0]) as tif:
        series = tif.series[0]
        img_shape = series.shape
        dtype = series.dtype
    
    print(f"Input image shape: {img_shape}, dimension order: {dim_order}, dtype: {dtype}")
    
    if dim_order not in ['TZYX', 'TYX']:
        raise ValueError(f"Expected dimension order 'TZYX' or 'TYX', but got '{dim_order}'")
    
    is_3d = len(img_shape) == 4 and dim_order == 'TZYX'
    
    if is_3d:
        time_points, z_slices, height, width = img_shape
        merged_shape = (time_points, len(channels), z_slices, height, width)
    else:
        time_points, height, width = img_shape
        merged_shape = (time_points, len(channels), height, width)
    
    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging files'):
        merged_img = np.zeros(merged_shape, dtype=dtype)

        for c, channel in enumerate(channels):
            print(f"Processing channel: {channel}")
            with TiffFile(file_lists[channel][i]) as tif:
                if is_3d:
                    for t in tqdm(range(time_points), desc=f"Processing {channel}"):
                        merged_img[t,c,:,:,:] = tif.asarray(key=slice(t * z_slices, (t + 1) * z_slices))
                else:
                    merged_img[:,c,:,:] = tif.asarray()

        # Reorder to TZCYX for 3D or TCYX for 2D
        if is_3d:
            merged_img = np.moveaxis(merged_img, 1, 2)
        else:
            merged_img = np.moveaxis(merged_img, 1, 1)  # No need to move axis for 2D

        # Save the merged image
        output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0], '').lstrip('-'))
            
        print(f"Saving merged image to: {output_filename}")
        print(f"Merged image shape: {merged_img.shape}")

        imwrite(output_filename, merged_img, compression='zlib', imagej=True, metadata={'axes': 'TZCYX' if is_3d else 'TCYX'})
    
    print("Merging completed successfully.")

def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    dim_order = args.dim_order.upper()

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

    if 'T' in dim_order:
        merge_channels_time_series_cpu(file_lists, channels, dim_order, merged_dir)
    else:
        merge_channels_cpu(file_lists, channels, dim_order, merged_dir)

    print("Merged images saved in", merged_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

