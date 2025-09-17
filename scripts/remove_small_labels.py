import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Add tmidas to path
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image
from tmidas.processing.segmentation import filter_small_labels

def process_label_file(file_path, min_size, output_type):
    try:
        label_image = read_image(file_path)
        if label_image is None:
            return

        # Filter small labels using the utility function
        new_label_image = filter_small_labels(label_image, min_size, output_type)
        
        # Save the edited label image
        write_image(new_label_image, file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def remove_small_labels(folder_path, label_suffix, min_size, output_type, max_workers=None):
    files = [f for f in os.listdir(folder_path) if f.endswith(label_suffix)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_label_file, os.path.join(folder_path, file), min_size, output_type) for file in files]
        
        for _ in tqdm(as_completed(futures), total=len(files), desc=f"Removing labels smaller than {min_size}"):
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--label_suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    parser.add_argument("--min_size", type=float, default=250.0, help="Exclude small labels.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker threads.")
    parser.add_argument("--output_type", type=str, choices=['semantic', 'instance'], required=True, 
                        help="Specify whether the output should be semantic or instance segmentation.")

    return parser.parse_args()


def main():
    args = parse_args()
    remove_small_labels(args.input, args.label_suffix, args.min_size, args.output_type, args.max_workers)


if __name__ == "__main__":
    main()
