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
    parser.add_argument('--label_folders', nargs=2, type=str, required=True, help='Folder names of the two label folders. Example: "conditions labels"')
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

def find_matching_file(base_filename, file_list):
    """
    Find the corresponding file in the second folder based on the base filename.
    
    :param base_filename: Base filename to match
    :param file_list: List of files to search in
    :return: Matching file path or None if not found
    """
    base_name = os.path.basename(base_filename)
    # Extract the part of the filename without the folder-specific suffix
    parts = base_name.split('_')
    if len(parts) > 1:
        # Remove the last part (e.g., "_conditions.tif")
        base_part = '_'.join(parts[:-1])
    else:
        base_part = os.path.splitext(base_name)[0]
    
    # Look for files in the second list that contain the base part
    for file_path in file_list:
        if base_part in os.path.basename(file_path):
            return file_path
    
    return None

def colocalize(file_lists, label_folders):
    """
    Perform colocalization analysis between two label folders.
    :param file_lists: Dictionary containing lists of file paths for each folder.
    :param label_folders: List of folder names.
    :return: List of rows for the output CSV file.
    """
    csv_rows = []
    
    # Process files from the first folder
    files_c1 = file_lists[label_folders[0]]
    files_c2 = file_lists[label_folders[1]]
    
    for file_path in tqdm(files_c1, total=len(files_c1), desc="Processing images"):
        try:
            # Load the first image
            image_c1 = load_image(file_path)
            if image_c1 is None:
                continue
                
            print(f"Loaded image {file_path} with shape {image_c1.shape}")
            
            # Find the corresponding file in the second folder
            matching_file = find_matching_file(file_path, files_c2)
            
            if matching_file is None:
                print(f"No matching file found for {os.path.basename(file_path)} in {label_folders[1]} folder")
                continue
                
            # Load the second image
            image_c2 = load_image(matching_file)
            if image_c2 is None:
                continue
                
            print(f"Loaded matching image {matching_file} with shape {image_c2.shape}")
            
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
        raise ValueError("Exactly two label folders must be provided.")
    
    # Collect files matching patterns in both label folders
    file_lists = {
        folder: sorted(glob.glob(os.path.join(parent_dir, folder, pattern)))
        for folder, pattern in zip(label_folders, label_patterns)
    }
    
    # Log the number of files found in each folder
    for folder, files in file_lists.items():
        print(f"Found {len(files)} files in {folder} folder")
    
    # Perform colocalization analysis
    csv_rows = colocalize(file_lists, label_folders)
    
    # Save results to a CSV file in the parent directory
    output_csv = os.path.join(parent_dir, 'label_colocalization_results.csv')
    
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Filename', f"{label_folders[0]}_ID", f"{label_folders[1]}_ID"])
        writer.writerows(csv_rows)
    
    print(f"Colocalization results saved to {output_csv}")
    print(f"Total colocalized regions found: {len(csv_rows)}")

if __name__ == "__main__":
    main()