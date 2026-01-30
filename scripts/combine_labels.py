import os
import numpy as np
from tifffile import imread, imwrite
import argparse
import cupy as cp
import cupyx.scipy.ndimage as cpndi
from tqdm import tqdm
import pyclesperanto_prototype as cle

"""
Description: This script creates a new label image by merging two label images. 
This is done using the cle.combine_labels function from pyclesperanto_prototype.

The script reads two label images from the input folder, 
merges them, and saves the output images in the same folder.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Label image merger.')
    parser.add_argument('--input', type=str, help='path to the input folder containing label images.')
    parser.add_argument('--label1_tag', type=str, help='tag of first label images')
    parser.add_argument('--label2_tag', type=str, help='tag of second label images')
    parser.add_argument('--output_tag', type=str, help='tag of output images')
    return parser.parse_args()

args = parse_args()

filenames = [f.replace(args.label1_tag, '') for f in os.listdir(args.input) if f.endswith(args.label1_tag)]

def merge_labels_gpu(label1, label2):
    # Read images
    label1_img = cp.asarray(imread(label1))
    label2_img = cp.asarray(imread(label2))
    
    # Convert to float32
    label1_img = label1_img.astype(np.float32)
    label2_img = label2_img.astype(np.float32)
    
    # Create GPU images
    label1_gpu = cle.push(cp.asnumpy(label1_img))
    label2_gpu = cle.push(cp.asnumpy(label2_img))
    
    # Merge labels
    result_gpu = cle.create_like(label1_gpu)
    cle.combine_labels(label1_gpu, label2_gpu, result_gpu)
    
    # Pull result back to CPU
    result = cle.pull(result_gpu)
    
    return result.astype(np.uint32)  # Convert back to uint32



for idx, filename in enumerate(tqdm(filenames, total=len(filenames), desc="Processing images")):
    label1 = os.path.join(args.input, filename + args.label1_tag)
    label2 = os.path.join(args.input, filename + args.label2_tag)
    
    if cp.cuda.is_available():
        result = merge_labels_gpu(label1, label2)
        print('\nUsing GPU')
    else:
        print('\nGPU not available. This script requires GPU.')
        break
    
    imwrite(os.path.join(args.input, filename + args.output_tag), 
            result, compression='zlib')

print("Processing complete.")
