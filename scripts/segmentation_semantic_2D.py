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
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from skimage.measure import label
import cv2
import apoc


"""
Description: This script runs automatic semantic segmentation on 2D fluorescence images using a pre-trained PixelClassifier
or Otsu thresholding for 2D brightfield images.

"""


def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--image_type", type=str, required=True, help="Brightfield images? (y/n)")
    return parser.parse_args()

args = parse_args()
    
image_folder = args.input
image_type = args.image_type


cl_filename = os.path.join(os.environ['TMIDAS_PATH'], "models/PixelClassifier_Joao_brightfield.cl")
classifier = apoc.PixelClassifier(cl_filename)

def process_image(image_path, image_type):
    try:
        if image_type == "y":
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_labeled = classifier.predict(image) 
            image_labeled = np.array(image_labeled, dtype=np.uint64)
                    # relabel 1 to zero and 2 to 1
            image_labeled[image_labeled == 1] = 0
            image_labeled[image_labeled == 2] = 1

        else:
            image = imread(image_path)
            image_to = cle.push(image)
            image_to = cle.gaussian_blur(image_to, None, 2.0, 2.0, 0.0)
            image_to = cle.threshold_otsu(image_to)
            image_to = cle.exclude_small_labels(image_to,None, 1000.0)
            image_labeled = cle.pull(image_to)
            image_labeled[image_labeled > 0] = 1    # relabel to 0 and 1

        return image_labeled
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None





for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
    if not filename.endswith(".tif"):
        continue
    #print(f"Processing image: {filename}")
    labeled_image = process_image(os.path.join(image_folder, filename), image_type)
    if labeled_image is not None:
        tf.imwrite(os.path.join(image_folder, f"{filename[:-4]}_semantic_seg.tif"), labeled_image, compression='zlib')
        
  