import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from skimage.io import imread
from skimage import measure
from tifffile import imwrite

def load_image(image_path):
    try:
        return imread(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, filename):
    try:
        image_uint32 = image.astype(np.uint32)
        imwrite(filename, image_uint32, compression='zlib')
    except Exception as e:
        print(f"Error saving image {filename}: {e}")

def process_label_file(file_path, min_size):
    try:
        label_image = load_image(file_path)
        if label_image is None:
            return

        # Create a temporary image where each connected component has a unique ID
        temp_labels = measure.label(label_image > 0, connectivity=1)
        
        # Get properties of each object (connected component)
        props = measure.regionprops(temp_labels, intensity_image=label_image)
        
        # Create an empty array to store the new label image
        new_label_image = np.zeros_like(label_image, dtype=label_image.dtype)
        
        # Iterate over each object and decide whether to keep it
        for prop in props:
            if prop.area > min_size:
                original_label_id = int(prop.intensity_mean)
                new_label_image[temp_labels == prop.label] = original_label_id
        
        # Save the edited label image
        save_image(new_label_image, file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def remove_small_labels(folder_path, label_suffix, min_size, max_workers=None):
    files = [f for f in os.listdir(folder_path) if f.endswith(label_suffix)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_label_file, os.path.join(folder_path, file), min_size) for file in files]
        
        for _ in tqdm(as_completed(futures), total=len(files), desc=f"Removing labels smaller than {min_size}"):
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--label_suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    parser.add_argument("--min_size", type=float, default=250.0, help="Exclude small labels.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker threads.")

    return parser.parse_args()

def main():
    args = parse_args()
    remove_small_labels(args.input, args.label_suffix, args.min_size, args.max_workers)

if __name__ == "__main__":
    main()
