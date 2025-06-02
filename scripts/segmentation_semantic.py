import os
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import pyclesperanto_prototype as cle
from tqdm import tqdm

"""
Description: This script runs automatic semantic segmentation on 2D or 3D images using Otsu thresholding or manual threshold.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--threshold", type=int, default=None, help="Enter an intensity threshold value within the range 1-255 if you want to define it yourself or enter 0 to use gauss-otsu thresholding.")
    parser.add_argument("--gamma", type=float, default="1.0", help="Gamma value for gamma correction. Default: 1.0 / no correction.")
    parser.add_argument("--sigma", type=float, default="0.0", help="Sigma value for Gaussian blur (default: 0 / no blur).")
    return parser.parse_args()

args = parse_args()

image_folder = args.input
threshold = args.threshold
GAMMA = args.gamma
SIGMA = args.sigma

def process_image(image_path):
    try:
        image = imread(image_path)
        image_to = cle.push(image)

        # Apply gamma correction only if gamma is set and not 1.0
        if GAMMA and GAMMA != "1.0":
            image_to = cle.gamma_correction(image_to, None, GAMMA)
        # Apply Gaussian blur only if sigma > 0
        if SIGMA and SIGMA > 0:
            image_to = cle.gaussian_blur(image_to, None, SIGMA, SIGMA, 0.0)

        if threshold == 0:
            image_to = cle.threshold_otsu(image_to)
        else:
            image_to = cle.greater_or_equal_constant(image_to, None, threshold)
        image_labeled = cle.pull(image_to)
        image_labeled[image_labeled > 0] = 1    # relabel to 0 and 1
        return image_labeled.astype(np.uint32)
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
    if not filename.endswith(".tif"):
        continue
    labeled_image = process_image(os.path.join(image_folder, filename))
    if labeled_image is not None:
        tf.imwrite(os.path.join(image_folder, f"{filename[:-4]}_semantic_seg.tif"), labeled_image, compression='zlib')
