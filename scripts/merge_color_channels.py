# script to batch merge channels
# each color channel is represented by a separate folder of images

import os
import argparse
import glob
from tqdm import tqdm
from skimage.io import imread
from tifffile import imwrite, TiffFile
# import cupy as cp
import numpy as np
import sys
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Batch merge channels')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--dim_order', type=str, default='XYC', help='Dimension order of the input images.')
    # parser.add_argument('--gpu', type=str, choices=['y', 'n'], required=True, help='Use GPU for processing y/n')
    return parser.parse_args()

def merge_channels_cpu(file_lists, channels, dim_order):
    # Merge the channels
    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging channels'):

        # loop through the channels
        for channel in channels:
            # load the image
            img = imread(file_lists[channel][i])

            if channel == channels[0]:
                print("\nCheck if image shape corresponds to the dim order that you have given:")
                print(f"Image shape: {img.shape}, dimension order: {dim_order}\n")
                # Determine if the image is 2D or 3D
                is_3d = len(img.shape) == 3 and 'Z' in dim_order

            if is_3d:
                if dim_order != 'ZYX':
                    img = np.transpose(img, transpose_order)
                # make empty merged image
                depth, height, width = img.shape
                merged_img = np.zeros((depth, height, width, 0), dtype=np.uint16)
                merged_img = np.concatenate((merged_img, img[..., np.newaxis]), axis=-1)
            else:
                if dim_order != 'YX':
                    img = np.transpose(img, transpose_order)
                # make empty merged image
                height, width = img.shape
                merged_img = np.zeros((height, width, 0), dtype=np.uint16)
                merged_img = np.dstack((merged_img, img))
        
        # save the merged image
        output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0],''))
        imwrite(output_filename, merged_img, compression='zlib')






def merge_channels_time_series_cpu(file_lists, channels, dim_order, merged_dir):
    # Merge the channels
    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging files'):
        # need to get image shape w/o loading the image
        with TiffFile(file_lists[channels[0]][i]) as tif:
            img_shape_c0 = tif.series[0].shape
            print("\nCheck if image shape corresponds to the dim order that you have given:")
            print(f"Image shape: {img_shape_c0}, dimension order: {dim_order}\n")
            time = img_shape_c0[dim_order.index('T')]

        # check if shape of all images is the same
        img_shapes = []
        for channel in channels:
            with TiffFile(file_lists[channel][i]) as tif:
                img_shapes.append(tif.series[0].shape)

        if len(set(img_shapes)) > 1:
            raise ValueError("All images must have the same shape.")

        # Determine if the image is 2D or 3D
        is_3d = len(img_shape_c0) >= 3 and 'Z' in dim_order

        if is_3d:
            if dim_order != 'TZYX':
                transpose_order = [dim_order.index(d) for d in 'TZYX']
        else:  # 2D case
            if dim_order != 'TYX':
                transpose_order = [dim_order.index(d) for d in 'TYX']

        # Pre-allocate the array for merged channels
        merged_shape = (*img_shape_c0, len(channels))
        merged_time_points = np.zeros(merged_shape, dtype=np.uint16)

        for t in tqdm(range(time), total=time, desc="Processing time points"):
            for c, channel in enumerate(channels):
                img = imread(file_lists[channel][i])
                img = np.array(img)
                if is_3d:
                    if dim_order != 'TZYX':
                        img = img.transpose(transpose_order)
                else:
                    if dim_order != 'TYX':
                        img = img.transpose(transpose_order)
                merged_time_points[..., c] = img

        # save the merged image
        output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0],''))
        imwrite(output_filename, merged_time_points, compression='zlib')

    print("Merging completed successfully.")











# def merge_channels_gpu(file_lists, channels, dim_order):
#     # Merge the channels
#     for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging channels'):
#         # create an empty merged image
#         img = imread(file_lists[channels[0]][i])

#         print("\nCheck if image shape corresponds to the dim order that you have given:")
#         print(f"Image shape: {img.shape}, dimension order: {dim_order}\n")

#         # Determine if the image is 2D or 3D
#         is_3d = len(img.shape) == 3 and 'Z' in dim_order

#         if is_3d:
#             if dim_order != 'ZYX':
#                 transpose_order = [dim_order.index(d) for d in 'ZYX']
#                 img = cp.array(img).transpose(transpose_order)
#             depth, height, width = img.shape
#             merged_img = cp.zeros((depth, height, width, 0), dtype=cp.uint16)
#         else:  # 2D case
#             if dim_order != 'YX':
#                 transpose_order = [dim_order.index(d) for d in 'YX']
#                 img = cp.array(img).transpose(transpose_order)
#             height, width = img.shape
#             merged_img = cp.zeros((height, width, 0), dtype=cp.uint16)

#         # loop through the channels
#         for channel in channels:
#             # load the image
#             img = imread(file_lists[channel][i])
#             img = cp.array(img)
#             if is_3d:
#                 if dim_order != 'ZYX':
#                     img = img.transpose(transpose_order)
#                 merged_img = cp.concatenate((merged_img, img[..., cp.newaxis]), axis=-1)
#             else:
#                 if dim_order != 'YX':
#                     img = img.transpose(transpose_order)
#                 merged_img = cp.dstack((merged_img, img))

#         # change dim order to CYX or CZYX
#         merged_img = cp.moveaxis(merged_img, source=-1, destination=0)
        
#         # save the merged image
#         output_filename = os.path.join(merged_dir, os.path.basename(file_lists[channels[0]][i]).replace(channels[0],''))
#         imwrite(output_filename, cp.asnumpy(merged_img), compression='zlib')

#     print("Merged images saved in", merged_dir)

def main():
    args = parse_args()

    parent_dir = args.input + '/'
    channels = [c.upper() for c in args.channels]
    dim_order = args.dim_order.upper()

    # Get a list of files for each channel
    file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel + '/*.tif'))) for channel in channels}

    # exclude files with pattern _labels.tif
    file_lists = {channel: [f for f in file_lists[channel] if not f.endswith('_labels.tif')] for channel in file_lists}

    # sort the files
    file_lists = {channel: sorted(file_lists[channel]) for channel in file_lists}

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    print("Number of images in each channel:")
    for channel in file_lists:
        print(f"{channel}: {len(file_lists[channel])}")

    # Check if all channels have the same number of files
    if len(set(len(file_lists[channel]) for channel in channels)) > 1:
        raise ValueError("All channels must have the same number of files.")

    # Create a new folder to save the merged images
    global merged_dir
    merged_dir = os.path.join(parent_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    # if args.gpu == 'y':
    #     merge_channels_gpu(file_lists, channels, dim_order)
    # elif args.gpu == 'n':
    #     merge_channels_cpu(file_lists, channels, dim_order)
    # else:
    #     raise ValueError("Invalid value for --gpu. Use y or n.")

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
