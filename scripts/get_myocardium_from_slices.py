#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from skimage.io import imread

import tifffile as tf
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
from skimage.measure import label



# Argument Parsing
parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
args = parser.parse_args()

#image_path = "BrdU/spon2bKO_7dpi_1 -3 - FITC_8bit.tif"



# Define utility functions

def calculate_threshold(image):
    
    gray_areas = image[image > 0]
    INTENSITY_THRESHOLD = np.percentile(gray_areas, 25) + np.mean(gray_areas)
    
    return INTENSITY_THRESHOLD


# get label id of largest object using regionprops


def process_image(image_path):
    try:

        image = imread(image_path)
        INTENSITY_THRESHOLD = calculate_threshold(image)
        #image = cle.push(image)
        #image_b = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
        #image_to = cle.greater_or_equal_constant(image,None,INTENSITY_THRESHOLD)
        image_to = nsbatwm.threshold_li(image)
        image_to = label(image_to) # much faster for large images than cle.connected_components_labeling_box 
        image_to = cle.push(image_to)
        image_to = cle.exclude_small_labels(image_to,None, 1000)
        image_labeled = cle.pull(image_to)
        image_labeled = np.array(image_labeled, dtype=np.uint64)
        # set all nonzero values to 1
        image_labeled[image_labeled > 0] = 1    

        return image_labeled
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Process images
image_folder = os.path.join(args.input)
for filename in os.listdir(image_folder):
    if not filename.endswith(".tif"):
        continue
    print(f"Processing image: {filename}")
    labeled_image = process_image(os.path.join(image_folder, filename))
    if labeled_image is not None:
        tf.imwrite(os.path.join(image_folder, f"{filename[:-4]}_labels.tif"), labeled_image, compression='zlib')
        
        
        
