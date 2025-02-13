import os
import glob
import csv
from skimage.io import imread
import argparse
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

"""
This Python script calculates the colocalization of Regions of Interest (ROIs)
between two labeled images using bounding box overlap. The results, including colocalized
label IDs, are saved in a CSV file for easy analysis.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of labeled images.')
    parser.add_argument('--parent_folder', type=str, required=True, help='Path to the parent folder containing the label folders.')
    parser.add_argument('--label_folders', nargs=2, type=str, required=True, help='Folder names of the two label label_folders. Example: "conditions labels"')
    parser.add_argument('--label_patterns', nargs=2, type=str, required=True, help='Label patterns for each folder. Example: "*_conditions.tif *_labels.tif"')
    return parser.parse_args()

def load_image(file_path):
    """
    Load an image using skimage.io and return it as a NumPy array.
    """
    try:
        return imread(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def bounding_box_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    :param box1: Bounding box of the first region (min_row, min_col, max_row, max_col).
    :param box2: Bounding box of the second region (min_row, min_col, max_row, max_col).
    :return: True if the bounding boxes overlap; False otherwise.
    """
    return not (
        box1[3] <= box2[1] or  # Left edge of box1 is to the right of box2
        box1[1] >= box2[3] or  # Right edge of box1 is to the left of box2
        box1[2] <= box2[0] or  # Bottom edge of box1 is above box2
        box1[0] >= box2[2]     # Top edge of box1 is below box2
    )

def colocalize(file_lists, label_folders):
    """
    Perform colocalization analysis between two label_folders.
    :param file_lists: Dictionary containing lists of file paths for each folder.
    :param label_folders: List of folder names.
    :return: List of rows for the output CSV file.
    """
    csv_rows = []
    
    # Process files from the first folder
    file_paths = file_lists[label_folders[0]]
    
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Processing images"):
        try:
            # Load corresponding images from both label_folders
            image_c1 = load_image(file_path)
            print(f"Loaded image {file_path} with shape {image_c1.shape}")
            image_c2 = load_image(file_lists[label_folders[1]][file_paths.index(file_path)])
            
            if image_c1 is None or image_c2 is None:
                continue
            
            # Get region properties for both images
            props_c1 = regionprops(image_c1.astype(np.int32))
            props_c2 = regionprops(image_c2.astype(np.int32))
            
            # Check for bounding box overlaps between regions in both images
            for prop_c1 in props_c1:
                for prop_c2 in props_c2:
                    if bounding_box_overlap(prop_c1.bbox, prop_c2.bbox):
                        csv_rows.append([os.path.basename(file_path), prop_c1.label, prop_c2.label])
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return csv_rows

def main():
    args = parse_args()
    
    parent_dir = args.parent_folder
    label_folders = args.label_folders
    label_patterns = args.label_patterns
    
    if len(label_folders) != 2:
        raise ValueError("Exactly two label_folders must be provided.")
    
    # Collect files matching patterns in both label_folders
    file_lists = {
        folder: sorted(glob.glob(os.path.join(parent_dir, folder, pattern)))
        for folder, pattern in zip(label_folders, label_patterns)
    }
    
    # Perform colocalization analysis
    csv_rows = colocalize(file_lists, label_folders)
    
    # Save results to a CSV file in the parent directory
    output_csv = os.path.join(parent_dir, 'colocalization_results.csv')
    
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Filename', f"{label_folders[0]}_ID", f"{label_folders[1]}_ID"])
        writer.writerows(csv_rows)
    
    print(f"Colocalization results saved to {output_csv}")

if __name__ == "__main__":
    main()
