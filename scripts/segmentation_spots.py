import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle
from tqdm import tqdm

"""
Description: This script runs automatic instance segmentation on 2D or 3D images of bright spots.

"""



def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value for gamma correction.")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalize using min-max scaling? (yes/no)")
    parser.add_argument("--use_filters", type=bool, default=True, help="Use filters for user-defined segmentation? (yes/no)")
    parser.add_argument("--intensity_threshold", type=float, default=None, help="Intensity threshold for image segmentation.")
    parser.add_argument("--dim_order", type=str, default="YX", help="Dimension order of the input images.)")
    return parser.parse_args()

args = parse_args()

image_folder = args.input
dim_order = args.dim_order
SIZE_THRESHOLD = 100.0  # square pixels
GAMMA = args.gamma
use_filters = args.use_filters
normalize = args.normalize

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def normalize_to_uint8(image):
    imin, imax = np.min(image), np.max(image)
    if imax > imin:  # Avoid division by zero
        norm_img = (image - imin) * 255.0 / (imax - imin)
    else:
        norm_img = np.zeros_like(image)
    return norm_img.astype(np.uint8)


def process_image(image_path, dim_order):
    """Process a single image and return labeled image."""
    try:
        image = imread(image_path)
        if normalize:
            image = normalize_to_uint8(image)
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
                
        if GAMMA != 1.0:
            image = cle.gamma_correction(image, None, GAMMA)
        if use_filters:
            image = cle.gaussian_blur(image, None, 2.0, 2.0, 0.0)
        if args.intensity_threshold is not None:
            intensity_threshold = args.intensity_threshold
            if intensity_threshold == 0:
                label_image = cle.threshold_otsu(image)
            elif intensity_threshold > 0:
                label_image = cle.greater_or_equal_constant(image, None, intensity_threshold)
        else:
            # error handling for the case when intensity threshold is not provided
            print("Please provide an intensity threshold value.")
            return None
        image_l = cle.connected_components_labeling_box(label_image)


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

            if GAMMA != 1.0:
                img_t = cle.gamma_correction(img_t, None, GAMMA)
            if use_filters:
                img_t = cle.gaussian_blur(img_t, None, 2.0, 2.0, 0.0)
            if args.intensity_threshold is not None:
                intensity_threshold = args.intensity_threshold
                if intensity_threshold == 0:
                    label_image = cle.threshold_otsu(img_t)
                elif intensity_threshold > 0:
                    label_image = cle.greater_or_equal_constant(img_t, None, intensity_threshold)
            else:
                # error handling for the case when intensity threshold is not provided
                print("Please provide an intensity threshold value.")
                return None
            img_l = cle.connected_components_labeling_box(label_image)

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
