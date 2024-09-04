import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle
from tqdm import tqdm
# import torch

"""
Description: This script runs automatic instance segmentation on 2D or 3D images of bright spots.

"""



def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--bg", type=int, choices=[1, 2], required=True, help="Background type (1 for dark or 2 for tissue).")
    # allow manual choice of intensity threshold
    parser.add_argument("--intensity_threshold", type=float, default=None, help="Intensity threshold for image segmentation.")
    parser.add_argument("--dim_order", type=str, default="YX", help="Dimension order of the input images.)")
    return parser.parse_args()

args = parse_args()

image_folder = args.input
dim_order = args.dim_order
SIZE_THRESHOLD = 100.0  # square pixels
BG = args.bg


def calculate_threshold(image):
    """Calculate intensity threshold for image segmentation."""
    gray_areas = image[image > 0]
    #gray_areas = image.flatten()[np.flatnonzero(image)]

    intensity_threshold = np.percentile(gray_areas, 75) + np.mean(gray_areas)
    return intensity_threshold

def process_image(image_path, dim_order):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
        intensity_threshold = None  # Initialize intensity_threshold with a default value
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {image.shape}, dimension order: {dim_order}")
        print("\n")
        # Determine if the image is 2D or 3D
        is_3d = len(image.shape) == 3 and 'Z' in dim_order
        if is_3d:
            if dim_order != 'ZYX':
                transpose_order = [dim_order.index(d) for d in 'ZYX']
                image = np.transpose(image, transpose_order)
        else:  # 2D case
            if dim_order != 'YX':
                transpose_order = [dim_order.index(d) for d in 'YX']
                image = np.transpose(image, transpose_order)
                
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


def process_time_series_image(image_path, dim_order):
    """Process a time series image and return labeled image."""
    try:
        image = imread(image_path)
        # print value of each dimension
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {image.shape}, dimension order: {dim_order}")
        print("\n")
        # if dim order is not TZYX, then transpose the image
        # Determine if the image is 2D or 3D
        is_3d = len(image.shape) == 4 and 'Z' in dim_order
        
        if is_3d:
            if dim_order != 'TZYX':
                transpose_order = [dim_order.index(d) for d in 'TZYX']
                image = np.transpose(image, transpose_order)
        else:  # 2D case
            if dim_order != 'TYX':
                transpose_order = [dim_order.index(d) for d in 'TYX']
                image = np.transpose(image, transpose_order)

        # Pre-allocate the array for labeled time points
        labeled_time_points = np.zeros(image.shape, dtype=np.uint32)

        for t in tqdm(range(image.shape[0]), total=image.shape[0], desc="Processing time points"):
            # Extract the current time point
            img_t = np.take(image, t, axis=0)
            intensity_threshold = None  # Initialize intensity_threshold with a default value

            if BG == 1:
                if args.intensity_threshold is not None:
                    intensity_threshold = args.intensity_threshold
                    print(f"Using user-defined intensity threshold: {intensity_threshold}")
                else:
                    intensity_threshold = calculate_threshold(img_t)
                    print(f"Calculated intensity threshold: {intensity_threshold}")
                img = cle.top_hat_box(img_t, None, 10.0, 10.0, 0.0)
                img = cle.gaussian_blur(img, None, 1.0, 1.0, 0.0)
                img_to = cle.greater_or_equal_constant(img, None, intensity_threshold)
                img_l = cle.connected_components_labeling_box(img_to)
                print("Segmenting bright spots with tissue background")
            elif BG == 2:
                img_thb = cle.top_hat_box(img_t, None, 10.0, 10.0, 0.0)
                img_l = cle.gauss_otsu_labeling(img_thb, None, 1.0)
                print("Segmenting bright spots with dark background")

            labeled_time_points[t] = cle.pull(img_l)

        return labeled_time_points
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
        if 'T' in dim_order:
            labeled_image = process_time_series_image(os.path.join(image_folder, filename), dim_order)
        else:
            labeled_image = process_image(os.path.join(image_folder, filename), dim_order)
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            #tf.imwrite(output_path, labeled_image, compression='zlib')
            save_image(labeled_image, output_path)
            del labeled_image

if __name__ == "__main__":
    main()
