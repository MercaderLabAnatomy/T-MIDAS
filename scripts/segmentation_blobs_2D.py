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
    parser.add_argument("--sigma", type=float, default=1.0, help="Defines the sigma for the gauss-otsu-labeling.")
    return parser.parse_args()

args = parse_args()


LOWER_THRESHOLD = args.exclude_small
UPPER_THRESHOLD = args.exclude_large

def process_image(image_path):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
        image_to = cle.gauss_otsu_labeling(image, None, args.sigma)
        image_labeled = cle.connected_components_labeling_box(image_to)
        image_labeled = cle.exclude_small_labels(image_labeled, None, LOWER_THRESHOLD)
        image_labeled = cle.exclude_large_labels(image_labeled, None, UPPER_THRESHOLD)
        image_labeled = cle.closing_labels(image_labeled, None, 10.0)
        image_labeled = nsitk.binary_fill_holes(image_labeled)
        #image_S = nsbatwm.split_touching_objects(image_l, 9.0)
        #image_labeled = cle.connected_components_labeling_box(image_S)
        
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
    image_folder = os.path.join(args.input)
    for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
        if not filename.endswith(".tif"):
            continue
        #print(f"Processing image: {filename}")
        labeled_image = process_image(os.path.join(image_folder, filename))
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            #tf.imwrite(output_path, labeled_image, compression='zlib')
            save_image(labeled_image, output_path)

if __name__ == "__main__":
    main()
