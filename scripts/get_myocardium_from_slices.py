#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import tifffile as tf
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
from skimage.measure import label
import cv2

# Argument Parsing
parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--image_type", type=str, required=True, help="Brightfield images? (y/n)")
args = parser.parse_args()
image_folder = args.input
image_type = args.image_type


device = cle.get_device()
cle.select_device(device.name)

def process_image(image_path, image_type):
    try:
        if image_type == "y":
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_not(image)
        else:
            image = imread(image_path)

        image_to = cle.push(image)
        image_to = cle.gaussian_blur(image_to, None, 2.0, 2.0, 0.0)
        image_to = cle.threshold_otsu(image_to)
        image_to = cle.exclude_small_labels(image_to,None, 1000.0)
        image_labeled = cle.pull(image_to)
        image_labeled = np.array(image_labeled, dtype=np.uint64)
        image_labeled[image_labeled > 0] = 1    # relabel to 0 and 1

        return image_labeled
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

for filename in os.listdir(image_folder):
    if not filename.endswith(".tif"):
        continue
    print(f"Processing image: {filename}")
    labeled_image = process_image(os.path.join(image_folder, filename), image_type)
    if labeled_image is not None:
        tf.imwrite(os.path.join(image_folder, f"{filename[:-4]}_labels.tif"), labeled_image, compression='zlib')
        
  