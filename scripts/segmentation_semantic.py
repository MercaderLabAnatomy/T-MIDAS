import os
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import pyclesperanto_prototype as cle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--threshold", type=int, default=None, help="Threshold value (1-255) or 0 for Otsu.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value (default: 1.0).")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian blur sigma (default: 0.0).")
    return parser.parse_args()

args = parse_args()

def process_image(image_path):
    try:
        image = imread(image_path)
        image_to = cle.push(image)

        if args.gamma != 1.0:
            image_to = cle.gamma_correction(image_to, None, args.gamma)
        if args.sigma > 0:
            image_to = cle.gaussian_blur(image_to, None, args.sigma, args.sigma, 0.0)

        if args.threshold == 0:
            image_to = cle.threshold_otsu(image_to)
        else:
            image_to = cle.greater_or_equal_constant(image_to, None, args.threshold)

        return cle.pull(image_to).astype(np.uint32)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Filter the list of files to process only valid images
file_list = [
    f for f in os.listdir(args.input)
    if f.endswith(".tif") and "semantic_seg" not in f
]

for filename in tqdm(file_list, desc="Processing images"):
    input_path = os.path.join(args.input, filename)
    labeled_image = process_image(input_path)
    if labeled_image is not None:
        base = os.path.splitext(filename)[0]
        params = f"t{args.threshold}_g{args.gamma:.2f}_s{args.sigma:.2f}"
        output_path = os.path.join(args.input, f"{base}_semantic_seg_{params}.tif")
        tf.imwrite(output_path, labeled_image, compression='zlib')
