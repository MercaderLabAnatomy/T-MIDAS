import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--exclude_small", type=float, default=250.0, help="Exclude small objects.")
    parser.add_argument("--exclude_large", type=float, default=50000.0, help="Exclude large objects.")
    parser.add_argument("--dim_order", type=str, default="YX", help="Dimension order of the input images.)")
    parser.add_argument("--threshold", type=int, default=None, help="Enter an intensity threshold value within in the range 1-255 if you want to define it yourself or enter 0 to use gauss-otsu thresholding.")
    return parser.parse_args()

args = parse_args()

dim_order = args.dim_order
threshold = args.threshold
SIGMA = 1.0
RADIUS = 10.0
LOWER_THRESHOLD = args.exclude_small
UPPER_THRESHOLD = args.exclude_large

def process_image(image_path, dim_order, threshold):
    """Process an image (single or time series) and return labeled image."""
    try:
        image = imread(image_path)
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {image.shape}, dimension order: {dim_order}")
        print("\n")

        # Determine if the image is a time series
        is_time_series = 'T' in dim_order
        is_3d = ('Z' in dim_order)

        if is_time_series:
            if is_3d and dim_order != 'TZYX':
                transpose_order = [dim_order.index(d) for d in 'TZYX']
                image = np.transpose(image, transpose_order)
            elif not is_3d and dim_order != 'TYX':
                transpose_order = [dim_order.index(d) for d in 'TYX']
                image = np.transpose(image, transpose_order)
        else:
            if is_3d and dim_order != 'ZYX':
                transpose_order = [dim_order.index(d) for d in 'ZYX']
                image = np.transpose(image, transpose_order)
            elif not is_3d and dim_order != 'YX':
                transpose_order = [dim_order.index(d) for d in 'YX']
                image = np.transpose(image, transpose_order)

        if is_time_series:
            labeled_time_points = np.zeros(image.shape, dtype=np.uint32)
            for t in tqdm(range(image.shape[0]), total=image.shape[0], desc="Processing time points"):
                img_t = np.take(image, t, axis=0)
                labeled_time_points[t] = process_single_image(img_t, is_3d, threshold)
            return labeled_time_points
        else:
            return process_single_image(image, is_3d, threshold)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def process_single_image(image, is_3d, threshold):
    """Process a single image slice and return labeled image."""
    if threshold == 0:
        if is_3d:
            image_to = cle.top_hat_box(image, None, RADIUS, RADIUS, RADIUS)
            image_to = cle.gauss_otsu_labeling(image_to, None, SIGMA)
        else:
            image_to = cle.top_hat_box(image, None, RADIUS, RADIUS, 0.0)
            image_to = cle.gauss_otsu_labeling(image_to, None, SIGMA)
    else:
        intensity_threshold = threshold
        print(f"Using user-defined intensity threshold: {intensity_threshold}")
        if is_3d:
            image_to = cle.top_hat_box(image, None, RADIUS, RADIUS, RADIUS)
            image_to = cle.gaussian_blur(image_to, None, SIGMA, SIGMA, SIGMA)
            image_to = cle.greater_or_equal_constant(image_to, None, intensity_threshold)
        else:
            image_to = cle.top_hat_box(image, None, RADIUS, RADIUS, 0.0)
            image_to = cle.gaussian_blur(image_to, None, SIGMA, SIGMA, 0.0)
            image_to = cle.greater_or_equal_constant(image_to, None, intensity_threshold)

    image_labeled = cle.connected_components_labeling_box(image_to)
    image_labeled = cle.exclude_small_labels(image_labeled, None, LOWER_THRESHOLD)
    image_labeled = cle.exclude_large_labels(image_labeled, None, UPPER_THRESHOLD)
    return image_labeled

def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

def main():
    image_folder = os.path.join(args.input)
    for filename in tqdm(os.listdir(image_folder), total=len(os.listdir(image_folder)), desc="Processing images"):
        if not filename.endswith(".tif"):
            continue
        labeled_image = process_image(os.path.join(image_folder, filename), dim_order, threshold)
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            save_image(labeled_image, output_path)

if __name__ == "__main__":
    main()
