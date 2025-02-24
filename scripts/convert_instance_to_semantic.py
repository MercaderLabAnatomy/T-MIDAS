import os
from skimage.io import imread
import argparse
from tifffile import imwrite
import numpy as np
from tqdm import tqdm


def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

def load_and_edit_labels(folder_path, label_suffix):
    """
    Load label images from a folder and set all non-zero labels to 1.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter files based on the label suffix
    label_files = [file for file in files if file.endswith(label_suffix)]

    # Load and edit each label file
    for file in tqdm(label_files, desc="Processing label images"):
        file_path = os.path.join(folder_path, file)
        label_image = imread(file_path)

        # Set all non-zero labels to 1
        label_image[label_image != 0] = 1

        # Save the edited labels
        save_image(label_image, file_path)

    print(f"All label files in {folder_path} have been processed.")



def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing with napari.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    return parser.parse_args()

def main():
    args = parse_args()
    load_and_edit_labels(args.input, args.suffix)

if __name__ == "__main__":
    main()
