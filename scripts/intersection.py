import os
from tifffile import imread, imwrite
import argparse
import cupy as cp
from tqdm import tqdm

"""
Description: This script creates a new image by intersecting two images. 
This is done by setting the pixel values of the original image to zero where the mask image is zero.

The script reads the mask and original images from the input folder, 
intersects them, and saves the output images in the same folder.

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Blob-based crops.')
    parser.add_argument('--input', type=str, help='path to the input folder containing intensity and label images.')
    parser.add_argument('--maskfiles', type=str, help='tag of label images')
    parser.add_argument('--intersectfiles', type=str, help='tag of intensity images')
    parser.add_argument('--output_tag', type=str, help='tag of output images')
    parser.add_argument('--save_as_label', action='store_true', help='Save as label image (uint32).')
    return parser.parse_args()

args = parse_args()

filenames = [f.replace(args.maskfiles, '') for f in os.listdir(args.input) if f.endswith(args.maskfiles)]

def intersection_cpu(mask, original):
    mask = imread(mask)
    original = imread(original)
    original[mask == 0] = 0
    return original

def intersection_gpu(mask, original):
    mask = cp.asarray(imread(mask))
    original = cp.asarray(imread(original))
    original[mask == 0] = 0
    return cp.asnumpy(original)

for idx, filename in enumerate(tqdm(filenames, total=len(filenames), desc="Processing images")):
    # print processing file  of total files
    mask = os.path.join(args.input, filename + args.maskfiles)
    original = os.path.join(args.input, filename + args.intersectfiles)
    
    if cp.cuda.is_available():
        result = intersection_gpu(mask, original)
        print('\nUsing GPU')
    else:
        result = intersection_cpu(mask, original)
        print('\nUsing CPU')
    
    if args.save_as_label:
        result = result.astype('uint32')
    
    imwrite(os.path.join(args.input, filename + args.output_tag), result, compression='zlib')
