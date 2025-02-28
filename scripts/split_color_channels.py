import os
import argparse
import glob
from tqdm import tqdm
from tifffile import imwrite, TiffFile
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Batch split channels')
    parser.add_argument('--input', type=str, required=True, help='Path to the folder containing multi-channel images.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Names of the color channels to split. Example: "TRITC DAPI FITC"')
    parser.add_argument('--time_steps', type=int, default=None,nargs='?', help='Number of time steps for timelapse images. Leave empty if not a timelapse.')

    return parser.parse_args()

def infer_dimension_order(shape, num_channels, time_steps):
    dim_order = ''
    if len(shape) == 5:
        dim_order = 'TZCYX'
    elif len(shape) == 4:
        if shape[0] == num_channels:
            dim_order = 'CZYX'
        elif time_steps and shape[0] == time_steps:
            dim_order = 'TCYX'
        else:
            dim_order = 'ZCYX'
    elif len(shape) == 3:
        if shape[0] == num_channels:
            dim_order = 'CYX'
        elif shape[2] == num_channels:
            dim_order = 'YXC'
        elif time_steps and shape[0] == time_steps:
            dim_order = 'TYX'
        else:
            raise ValueError(f"Unable to infer dimension order for shape {shape}")
    else:
        raise ValueError(f"Unsupported image shape: {shape}")
    
    return dim_order

def split_channels_cpu(file_list, channels, time_steps, output_dir):
    num_channels = len(channels)
    
    for file_path in tqdm(file_list, desc='Splitting files'):
        with TiffFile(file_path) as tif:
            img = tif.asarray()
            metadata = tif.imagej_metadata

        # Try to get dimension order from metadata
        dim_order = metadata.get('axes', '') if metadata else ''
        
        if not dim_order:
            # Infer dimension order from shape and user info
            dim_order = infer_dimension_order(img.shape, num_channels, time_steps)

        print(f"Input image shape: {img.shape}, detected dimension order: {dim_order}")

        # Find the channel axis
        channel_axis = dim_order.index('C')

        if img.shape[channel_axis] != num_channels:
            raise ValueError(f"Number of channels in the image ({img.shape[channel_axis]}) does not match the number of provided channel names ({num_channels})")

        for i, channel in enumerate(channels):
            channel_dir = os.path.join(output_dir, channel)
            os.makedirs(channel_dir, exist_ok=True)

            # Extract the channel
            channel_img = np.take(img, i, axis=channel_axis)

            # Remove the channel dimension from the dimension order
            output_dim_order = dim_order.replace('C', '')

            output_filename = os.path.join(channel_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_{channel}.tif")
            imwrite(output_filename, channel_img, compression='zlib', imagej=True, metadata={'axes': output_dim_order})

    print("Splitting completed successfully.")

def main():
    args = parse_args()
    input_dir = args.input
    channels = [c.upper() for c in args.channels]
    time_steps = args.time_steps

    file_list = sorted(glob.glob(os.path.join(input_dir, '*.tif')))

    if len(file_list) == 0:
        raise ValueError(f"No .tif files found in the input directory: {input_dir}")

    print(f"Number of images to process: {len(file_list)}")

    output_dir = os.path.join(input_dir, 'split_channels')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_channels_cpu(file_list, channels, time_steps, output_dir)

    print("Split images saved in", output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
