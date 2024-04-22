# this python script employs cupy to split the color channels of an image
# user input is the folder containing the images to be split, as well as the dimension order of the images and the names of the color channel output folders


import os
import cupy as cp
import argparse
import sys
from skimage.io import imread
import numpy as np
from tqdm import tqdm
from tifffile import imwrite


def parse_args():
    parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the multicolor images.')
    parser.add_argument('--dim_order', type=str, help='Dimension order of the images (example: XYZCT).')
    parser.add_argument('--channel_names', type=str, nargs='+', help='Names of the color channels (example: FITC DAPI TRITC).')
    return parser.parse_args()

args = parse_args()


folder = args.input


def save_image(image, filename):
    # Convert image data type to uint32 before saving
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

def split_color_channels(folder, dim_order, channel_names):
    # get the list of files in the folder
    files = [file for file in os.listdir(folder) if file.endswith('.tif')]
    # loop through the files
    for file in tqdm(files):
        # load the image
        img = imread(os.path.join(folder, file))
        # convert the numpy array to a cupy array
        img = cp.array(img)
        # split the color channels
        channels = [img[:,:,i] for i in range(img.shape[dim_order.index('C')])]
        # save the color channels
        for i, channel in enumerate(channels):
            channel = cp.asnumpy(channel)
            save_image(channel, os.path.join(folder, f"{file[:-4]}_{channel_names[i]}.tif"))




if __name__ == '__main__':
    args = parser.parse_args()
    split_color_channels(args.input, args.dim_order, args.channel_names)
