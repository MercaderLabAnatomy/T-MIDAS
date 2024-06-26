import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--bg", type=int, choices=[1, 2], required=True, help="Background type (1 for dark or 2 for tissue).")
    # allow manual choice of intensity threshold
    parser.add_argument("--intensity_threshold", type=float, default=None, help="Intensity threshold for image segmentation.")
    return parser.parse_args()

args = parse_args()

image_folder = args.input
SIZE_THRESHOLD = 100.0  # square pixels
BG = args.bg


def calculate_threshold(image):
    """Calculate intensity threshold for image segmentation."""
    gray_areas = image[image > 0]
    #gray_areas = image.flatten()[np.flatnonzero(image)]

    intensity_threshold = np.percentile(gray_areas, 75) + np.mean(gray_areas)
    return intensity_threshold

def process_image(image_path):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
        intensity_threshold = None  # Initialize intensity_threshold with a default value

        if BG == 1:
            if args.intensity_threshold is not None:
                intensity_threshold = args.intensity_threshold
                print(f"Using user-defined intensity threshold: {intensity_threshold}")
            else:
                intensity_threshold = calculate_threshold(image)
                print(f"Calculated intensity threshold: {intensity_threshold}")
            image = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
            image_to = cle.greater_or_equal_constant(image, None, intensity_threshold)
            image_l = cle.connected_components_labeling_box(image_to)
            print("Segmenting bright spots with tissue background")
        elif BG == 2:
            image_thb = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image_l = cle.gauss_otsu_labeling(image_thb, None, 1.0)
            print("Segmenting bright spots with dark background")

        image_labeled = cle.pull(image_l)
        return image_labeled
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def save_image(image, filename):
    # Convert image data type to uint32 before saving
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')


def main():
    """Main function to process all images in the input directory."""

    for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
        if not filename.endswith(".tif"):
            continue
        #print(f"Processing image: {filename}")
        labeled_image = process_image(os.path.join(image_folder, filename))
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            #tf.imwrite(output_path, labeled_image, compression='zlib')
            save_image(labeled_image, output_path)
            del labeled_image

if __name__ == "__main__":
    main()
