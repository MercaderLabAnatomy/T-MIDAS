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
    # parser.add_argument("--sigma", type=float, default=1.0, help="Defines the sigma for the gauss-otsu-labeling.")
    parser.add_argument("--dim_order", type=str, default="YX", help="Dimension order of the input images.)")
    # add option to define threshold or use gauss-otsu
    parser.add_argument("--threshold", type=int, default=None, help="Enter an intensity threshold value within in the range 1-255 if you want to define it yourself or enter 0 to use gauss-otsu thresholding.")
    return parser.parse_args()

args = parse_args()

dim_order = args.dim_order
threshold = args.threshold
sigma = 1.0
LOWER_THRESHOLD = args.exclude_small
UPPER_THRESHOLD = args.exclude_large

def process_image(image_path,dim_order,threshold):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
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


        if threshold == 0:
            image = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image_to = cle.gauss_otsu_labeling(image, None, 1.0)
        else: 
            intensity_threshold = threshold
            print(f"Using user-defined intensity threshold: {intensity_threshold}")
            image = cle.top_hat_box(image, None, 10.0, 10.0, 0.0)
            image = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
            image_to = cle.greater_or_equal_constant(image, None, intensity_threshold)
            
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

def process_time_series_image(image_path,dim_order,threshold):
    """Process a time series image and return labeled image."""
    try:
        image = imread(image_path)
        # print value of each dimension
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {image.shape}, dimension order: {dim_order}")
        print("\n")
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
            if threshold == 1:
                intensity_threshold = threshold
                image_to = cle.greater_or_equal_constant(img_t, None, intensity_threshold)
                print(f"Using user-defined intensity threshold: {intensity_threshold}")
            else:
                image_to = cle.gauss_otsu_labeling(img_t, None, sigma)
            image_labeled = cle.connected_components_labeling_box(image_to)
            image_labeled = cle.exclude_small_labels(image_labeled, None, LOWER_THRESHOLD)
            image_labeled = cle.exclude_large_labels(image_labeled, None, UPPER_THRESHOLD)
            image_labeled = cle.closing_labels(image_labeled, None, 10.0)
            image_labeled = nsitk.binary_fill_holes(image_labeled)
            #image_S = nsbatwm.split_touching_objects(image_l, 9.0)
            #image_labeled = cle.connected_components_labeling_box(image_S)
            labeled_time_points[t] = image_labeled

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
    image_folder = os.path.join(args.input)
    for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
        if not filename.endswith(".tif"):
            continue
        #print(f"Processing image: {filename}")

        if 'T' in dim_order:
            labeled_image = process_time_series_image(os.path.join(image_folder, filename),dim_order,threshold)
        else:
            labeled_image = process_image(os.path.join(image_folder, filename),dim_order,threshold)
        
        
        
        if labeled_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_labels.tif")
            #tf.imwrite(output_path, labeled_image, compression='zlib')
            save_image(labeled_image, output_path)

if __name__ == "__main__":
    main()
