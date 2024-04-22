# script to batch merge channels
# each color channel is represented by a separate folder of images

import os
import argparse
import glob
from tqdm import tqdm
from skimage.io import imread
from tifffile import imwrite
import cupy as cp

def parse_args():
    parser = argparse.ArgumentParser(description='Batch merge channels')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    return parser.parse_args()

args = parse_args()





parent_dir = args.input + '/'
channels = [c.upper() for c in args.channels]


# Get a list of files for each channel
file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel + '/*.tif'))) for channel in channels}

# exclude files with pattern _labels.tif
file_lists = {channel: [f for f in file_lists[channel] if not f.endswith('_labels.tif')] for channel in file_lists}

# sort the files
file_lists = {channel: sorted(file_lists[channel]) for channel in file_lists}

if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")

print("Number of images in each channel:")
{print(channel, ":", len(file_lists[channel])) for channel in file_lists}

# show name of first file in each channel
{print(channel, ":", file_lists[channel][0]) for channel in file_lists}

# Create a new folder to save the merged images
merged_dir = parent_dir + 'merged/'
if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)

def merge_channels(file_lists, channels):

    # Merge the channels
    for i in tqdm(range(len(file_lists[channels[0]])), desc='Merging channels'):
        # create an empty merged image
        img = imread(file_lists[channels[0]][i])
        height, width = img.shape[:2]
        merged_img = cp.zeros((height, width, 0), dtype=cp.uint8)
        # loop through the channels
        for j, channel in enumerate(channels):
            # load the image
            img = imread(file_lists[channel][i])
            # convert the numpy array to a cupy array
            img = cp.array(img)
            # add the image to the merged image
            merged_img = cp.dstack((merged_img, img))
            # change dim order from XYC to CYX
        
        merged_img = cp.moveaxis(merged_img, source=-1, destination=0) # -1 means the last axis
        # save the merged image
        imwrite(os.path.join(merged_dir, 
                             os.path.basename(file_lists[channels[0]][i])), 
                cp.asnumpy(merged_img), compression='zlib')
        
        
    print("Merged images saved in", merged_dir)
    
merge_channels(file_lists, channels)

