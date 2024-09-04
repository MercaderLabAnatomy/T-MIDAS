import os
import argparse
import glob
from tqdm import tqdm
from tifffile import imread, imwrite, TiffFile
import numpy as np
import sys

"""
Description: This script splits multi-channel images into individual color channels.

It assumes that the images are in tif format and have the same dimensions.

"""



def parse_args():
    parser = argparse.ArgumentParser(description='Batch split channels')
    parser.add_argument('--input', type=str, required=True, help='Path to the folder containing multi-channel images.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Names of the color channels to split. Example: "TRITC DAPI FITC"')
    parser.add_argument('--dim_order', type=str, default='CYX', help='Dimension order of the input images.')
    return parser.parse_args()

def split_channels_cpu(file_list, channels, dim_order, output_dir):
    for file_path in tqdm(file_list, desc='Splitting files'):
        with TiffFile(file_path) as tif:
            img = tif.asarray()
            metadata = tif.imagej_metadata

        print(f"Input image shape: {img.shape}, dimension order: {dim_order}")

        if dim_order not in ['CYX', 'ZCYX', 'TCYX', 'TZCYX']:
            # reorder the image to the desired dimension order
            if len(img.shape) == 3 and dim_order != 'CYX':
                transpose_order = [dim_order.index(d) for d in 'CYX']
                img = np.transpose(img, transpose_order)
            elif len(img.shape) == 4 and dim_order != 'ZCYX':
                transpose_order = [dim_order.index(d) for d in 'ZCYX']
                img = np.transpose(img, transpose_order)
            elif len(img.shape) == 4 and dim_order != 'TCYX':
                transpose_order = [dim_order.index(d) for d in 'TCYX']
                img = np.transpose(img, transpose_order)
            elif len(img.shape) == 5 and dim_order != 'TZCYX':
                transpose_order = [dim_order.index(d) for d in 'TZCYX']
                img = np.transpose(img, transpose_order)
            else:
                raise ValueError(f"Expected dimensions 'CYX', 'ZCYX', 'TCYX', or 'TZCYX', but got '{dim_order}'")

            # raise ValueError(f"Expected dimension order 'CYX', 'ZCYX', 'TCYX', or 'TZCYX', but got '{dim_order}'")

        is_3d = 'Z' in dim_order
        is_time_series = 'T' in dim_order

        channel_axis = dim_order.index('C')
        num_channels = img.shape[channel_axis]

        if num_channels != len(channels):
            raise ValueError(f"Number of channels in the image ({num_channels}) does not match the number of provided channel names ({len(channels)})")

        for i, channel in enumerate(channels):
            channel_dir = os.path.join(output_dir, channel)
            os.makedirs(channel_dir, exist_ok=True)

            if is_time_series:
            
                if is_3d:
                    channel_img = img.take(i, axis=channel_axis)
                else:
                    channel_img = img.take(i, axis=channel_axis)
            else:
                if is_3d:
                    channel_img = img.take(i, axis=channel_axis)
                else:
                    channel_img = img.take(i, axis=channel_axis)

            output_filename = os.path.join(channel_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_{channel}.tif")
            imwrite(output_filename, channel_img, compression='zlib', imagej=True, metadata={'axes': dim_order.replace('C', '')})

    print("Splitting completed successfully.")

def main():
    args = parse_args()
    input_dir = args.input
    channels = [c.upper() for c in args.channels]
    dim_order = args.dim_order.upper()

    file_list = sorted(glob.glob(os.path.join(input_dir, '*.tif')))

    if len(file_list) == 0:
        raise ValueError(f"No .tif files found in the input directory: {input_dir}")

    print(f"Number of images to process: {len(file_list)}")

    output_dir = os.path.join(input_dir, 'split_channels')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_channels_cpu(file_list, channels, dim_order, output_dir)

    print("Split images saved in", output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
