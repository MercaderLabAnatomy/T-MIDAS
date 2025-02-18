import os
import napari
from skimage.io import imread
import argparse
from tqdm import tqdm
import pyclesperanto_prototype as cle  # version 0.24.5
import numpy as np
from tifffile import imwrite

def load_image(image_path):
    try:
        return imread(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None





def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')


def remove_small_labels(folder_path, label_suffix, min_size):
  
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter files based on the label suffix
    label_files = [file for file in files if file.endswith(label_suffix)]

    
    # Load and edit each label file
    for file in tqdm(label_files, desc="Removing labels smaller than " + str(min_size) + " from label images..."):
        file_path = os.path.join(folder_path, file)
        label_image = load_image(file_path)
        
        # Remove small labels
        label_image = cle.exclude_small_labels(label_image, None, min_size)

        # Save the edited label image
        save_image(label_image, file_path)



def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing with napari.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--label_suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    parser.add_argument("--min_size", type=float, default=250.0, help="Exclude small labels.")

    return parser.parse_args()

def main():
    args = parse_args()
    remove_small_labels(args.input, args.label_suffix, args.min_size)
if __name__ == "__main__":
    main()
