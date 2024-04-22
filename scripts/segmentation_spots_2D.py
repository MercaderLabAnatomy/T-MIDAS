import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--bg", type=int, choices=[1, 2], required=True, help="Background type (1 for dark or 2 for tissue).")
    return parser.parse_args()

args = parse_args()

SIZE_THRESHOLD = 100.0  # square pixels

def calculate_threshold(image):
    """Calculate intensity threshold for image segmentation."""
    gray_areas = image[image > 0]
    intensity_threshold = np.percentile(gray_areas, 75) + np.mean(gray_areas)
    return intensity_threshold

def process_image(image_path):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
        intensity_threshold = calculate_threshold(image)
        #image = cle.push(image)
        
        if args.bg == 1:
            image = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
            image = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image_to = cle.greater_or_equal_constant(image, None, intensity_threshold)
            print("Segmenting bright spots with tissue background")
        elif args.bg == 2:
            image = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image_to = cle.threshold_otsu(image, None)
            print("Segmenting bright spots with dark background")
        
        image_l = cle.connected_components_labeling_box(image_to)
        image_labeled = cle.exclude_small_labels(image_l, None, SIZE_THRESHOLD)
        image_labeled = cle.pull(image_labeled)
        #image_labeled = np.array(image_labeled, dtype=np.uint32)
        
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
    for filename in os.listdir(image_folder):
        if not filename.endswith(".tif"):
            continue
        print(f"Processing image: {filename}")
        labeled_image = process_image(os.path.join(image_folder, filename))
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            #tf.imwrite(output_path, labeled_image, compression='zlib')
            save_image(labeled_image, output_path)

if __name__ == "__main__":
    main()
