import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import pyclesperanto_prototype as cle
from skimage.measure import regionprops
from tqdm import tqdm

"""
This script creates a region of interest (ROI) image 
from a label image containing masks of the intact myocardium and injury regions.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Input: Folder with label images containing masks of intact myocardium and injury regions.")
    parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
    parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
    parser.add_argument("--intact_label_id", type=int, required=True, help="Label id of the intact myocardium.")
    parser.add_argument("--injury_label_id", type=int, required=True, help="Label id of the injury region.")
    parser.add_argument("--label_suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    return parser.parse_args()

def gpu_processing(array):
    label_image = cle.push(array)
    label_image = cle.merge_touching_labels(label_image)
    return label_image

def get_myocardium_wo_injury(image, intact_label_id):
    myocardium_wo_injury = np.copy(image)
    myocardium_wo_injury[myocardium_wo_injury != intact_label_id] = 0
    return myocardium_wo_injury

def get_injury(image, injury_label_id):
    injury = np.copy(image)
    injury[injury != injury_label_id] = 0
    return injury

def get_border_zone(injury, myocardium_wo_injury, border_zone_diameter_px):
    injury_dilated = cle.dilate_labels(injury, None, border_zone_diameter_px)
    border_zone = cle.binary_and(injury_dilated, myocardium_wo_injury)
    return cle.pull(border_zone)

def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

def process_image(filename, image_folder, intact_label_id, injury_label_id, border_zone_diameter_px, label_suffix):
    try:
        image = imread(os.path.join(image_folder, filename))
        myocardium_wo_injury = get_myocardium_wo_injury(image, intact_label_id)
        if myocardium_wo_injury is not None:
            injury = get_injury(image, injury_label_id)     
            border_zone = get_border_zone(injury, myocardium_wo_injury, border_zone_diameter_px)
            ROIs = np.zeros_like(image)
            ROIs[myocardium_wo_injury > 0] = 1
            ROIs[injury > 0] = 2
            ROIs[border_zone > 0] = 3
            save_image(ROIs, os.path.join(image_folder, filename.replace(label_suffix, "_ROIs.tif")))
        return f"Processed {filename} successfully"
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

def main():
    args = parse_args()
    PIXEL_RESOLUTION = args.pixel_resolution
    INJURY_LABEL_ID = args.injury_label_id
    INTACT_LABEL_ID = args.intact_label_id
    BORDER_ZONE_DIAMETER_UM = 100.0
    BORDER_ZONE_DIAMETER_PX = BORDER_ZONE_DIAMETER_UM / PIXEL_RESOLUTION
    image_folder = args.input
    label_suffix = args.label_suffix
    image_files = [f for f in os.listdir(image_folder) if f.endswith(label_suffix)]
    for filename in tqdm(image_files, desc="Processing images"):
        print(process_image(filename, image_folder, INTACT_LABEL_ID, INJURY_LABEL_ID, BORDER_ZONE_DIAMETER_PX, label_suffix))

if __name__ == "__main__":
    main()
