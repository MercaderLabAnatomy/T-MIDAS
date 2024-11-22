import os
import numpy as np
import argparse
import tifffile as tf
from skimage.measure import regionprops, label
import re
from collections import defaultdict
from tqdm import tqdm
import pyclesperanto_prototype as cle


"""
This script extracts elongated ROIs from Acquifer TIF files to crop them and save them as multi-color TIF files.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Extract elongated ROIs from Acquifer TIF files and save them as multi-color TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the TIF files.')
    parser.add_argument('--padding', type=int, default=50, help='Padding around the ROI (default: 50)')
    return parser.parse_args()

def safe_divide(a, b, default=0):
    return a / b if b != 0 else default

def get_roi(image, padding):
    if image is None or image.size == 0:
        return (0, 0, 0, 0)  # Return a default ROI if the image is None or empty

    # Normalize and convert to uint8
    image_min, image_max = np.min(image), np.max(image)
    if image_max > image_min:
        image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)  # All black image if min == max

    image1_gbp = cle.gaussian_blur(image, None, 10.0, 10.0, 0.0)
    image3_vsp = cle.variance_sphere(image1_gbp, None, 1.0, 1.0, 0.0)
    labels = cle.threshold_otsu(image3_vsp)

    # Get ROIs
    props = regionprops(labels, intensity_image=image)
    y0, x0, y1, x1 = props[0].bbox
    minr, minc = max(0, y0 - padding), max(0, x0 - padding)
    maxr, maxc = min(labels.shape[0], y1 + padding), min(labels.shape[1], x1 + padding)
    
    return (minr, minc, maxr - minr, maxc - minc)

def group_files(input_folder):
    pattern = r'^-([A-Z]\d+)--PO(\d+)' 
    """
    This pattern captures all files that correspond 
    to the same well and PO (position) 
    cf. https://www.acquifer.de/resources/metadata/

    This means that all groups will be cropped based 
    on the ROI (+padding) of the first CO1 (brightfield) 
    file found in the group.
    """
    files_grouped_by_well_and_position = defaultdict(list)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            match = re.match(pattern, filename)
            if match:
                key = f"{match.group(1)}--PO{match.group(2)}"
                files_grouped_by_well_and_position[key].append(filename)
    
    return files_grouped_by_well_and_position

def process_files(input_folder, output_dir, padding):
    files_grouped_by_well_and_position = group_files(input_folder)
    print(f"Found {len(files_grouped_by_well_and_position)} groups of files.")

    for group_key, files in tqdm(files_grouped_by_well_and_position.items(), desc="Processing file groups"):
        try:
            # Find the CO1 file in the group
            co1_file = next((f for f in files if '--CO1--' in f), None)
            if not co1_file:
                print(f"No CO1 file found for group {group_key}. Skipping.")
                continue

            # Load CO1 image and get ROI
            with tf.TiffFile(os.path.join(input_folder, co1_file)) as tif:
                co1_image = tif.asarray()
            roi = get_roi(co1_image, padding)

            # Process all files in the group
            for file in files:
                with tf.TiffFile(os.path.join(input_folder, file)) as tif:
                    channel_image = tif.asarray()
                
                # Crop and normalize
                y, x, h, w = roi
                cropped_image = channel_image[y:y+h, x:x+w]
                normalized_image = ((cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min()) * 65535).astype(np.uint16)
                
                # Save cropped image
                output_filename = f"{os.path.splitext(file)[0]}_cropped.tif"
                output_path = os.path.join(output_dir, output_filename)
                tf.imwrite(output_path, normalized_image, compression='zlib')
                print(f"Saved: {output_filename}")

        except Exception as e:
            print(f"Error processing group {group_key}: {str(e)}")

def main():
    args = parse_args()
    input_folder = args.input
    padding = args.padding
    output_dir = os.path.join(input_folder, "processed_tifs")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_files(input_folder, output_dir, padding)

if __name__ == "__main__":
    main()
