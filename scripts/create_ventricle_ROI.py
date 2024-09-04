#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
import pyclesperanto_prototype as cle  # version 0.24.2
from skimage.measure import regionprops
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

"""
Description: Create a region of interest (ROI) image containing the following regions:
- Myocardium
- Myocardium without injury
- Injury
- Border zone

The input images are label images containing the following labels:
- Intact myocardium (label id: INTACT_LABEL_ID)
- Injury region (label id: INJURY_LABEL_ID)
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Input: Folder with label images containing masks of intact myocardium and injury regions.")
    parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
    parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
    parser.add_argument("--intact_label_id", type=int, required=True, help="Label id of the intact myocardium.")
    parser.add_argument("--injury_label_id", type=int, required=True, help="Label id of the injury region.")
    return parser.parse_args()

args = parse_args()

PIXEL_RESOLUTION = args.pixel_resolution

INJURY_LABEL_ID = args.injury_label_id
INTACT_LABEL_ID = args.intact_label_id
BORDER_ZONE_DIAMETER_UM = 100.0
BORDER_ZONE_DIAMETER_PX = BORDER_ZONE_DIAMETER_UM / PIXEL_RESOLUTION
SMALL_LABELS_THRESHOLD = 10000.0

def gpu_processing(array):
    try:
        label_image = cle.push(array)
        label_image = cle.merge_touching_labels(label_image)
        return label_image
    except Exception as e:
        print(f"Error processing {image}: {str(e)} on the GPU.")
        return None

def get_largest_label(label_image):
    label_image = cle.connected_components_labeling_box(label_image)
    label_props = regionprops(label_image)
    areas = [region.area for region in label_props]
    max_area_label = np.argmax(areas) + 1 
    return cle.equal_constant(label_image, None, max_area_label)

def get_myocardium(image):
    try:
        myocardium = gpu_processing(image)
        myocardium = get_largest_label(myocardium)
        return cle.pull(myocardium)
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting myocardium.")
        return None

def get_myocardium_wo_injury(image):
    try:
        myocardium_wo_injury = np.copy(image)
        myocardium_wo_injury[myocardium_wo_injury != INTACT_LABEL_ID] = 0
        return myocardium_wo_injury
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting myocardium without injury.")
        return None

def get_injury(image):
    try:
        injury = np.copy(image)
        injury[injury != INJURY_LABEL_ID] = 0
        return injury
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting injury.")
        return None

def get_border_zone(injury, myocardium_wo_injury):
    try:
        injury_dilated = cle.dilate_labels(injury, None, BORDER_ZONE_DIAMETER_PX)
        border_zone = cle.binary_and(injury_dilated, myocardium_wo_injury)
        return cle.pull(border_zone)
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting border zone.")
        return None

def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

image_folder = os.path.join(args.input)

for filename in tqdm(os.listdir(image_folder), total=len(os.listdir(image_folder)), desc="Processing images"):
    if not filename.endswith("_labels.tif"):
        continue
    
    print(f"Processing image: {filename}")
    
    image = imread(os.path.join(image_folder, filename))
    
    myocardium = get_myocardium(image)
    myocardium_wo_injury = get_myocardium_wo_injury(image)
    
    if myocardium is not None and myocardium_wo_injury is not None:
        
        injury = get_injury(image)     
        border_zone = get_border_zone(injury, myocardium_wo_injury)
        ROIs = np.zeros_like(myocardium)
        ROIs[myocardium > 0] = 1
        ROIs[myocardium_wo_injury > 0] = 2
        ROIs[injury > 0] = 3
        ROIs[border_zone > 0] = 4
        
        save_image(ROIs, os.path.join(image_folder, filename.replace(".tif", "_labels.tif")))
