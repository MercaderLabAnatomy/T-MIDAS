#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
import pyclesperanto_prototype as cle
from skimage.measure import regionprops

# Argument Parsing
parser = argparse.ArgumentParser(description="Input: Folder with label images containing masks of intact ventricle and injury regions.")
parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
args = parser.parse_args()

EPICARDIUM_DIAMETER_UM = 15.0
EPICARDIUM_ADDITIONAL_DIAMETER_UM = 10.0
PIXEL_RESOLUTION = args.pixel_resolution # Slidescanner Hamamatsu S60 at 40x is 0.23 um/px
EPICARDIUM_DIAMETER_PX = EPICARDIUM_DIAMETER_UM / PIXEL_RESOLUTION
EPICARDIUM_ADDITIONAL_DIAMETER_PX = EPICARDIUM_ADDITIONAL_DIAMETER_UM / PIXEL_RESOLUTION
INTACT_VENTRICLE_REGION_LABEL_ID = 1
INJURY_LABEL_ID = 2
BORDER_ZONE_DIAMETER_UM = 100.0
BORDER_ZONE_DIAMETER_PX = BORDER_ZONE_DIAMETER_UM / PIXEL_RESOLUTION
SMALL_LABELS_THRESHOLD = 10000.0 # square pixels

# Define utility functions

def gpu_processing(array):
    try:
        label_image = cle.push(array)
        label_image = cle.merge_touching_labels(label_image)
        label_image = nsitk.binary_fill_holes(label_image)
        label_image = cle.connected_components_labeling_box(label_image)
        label_image = cle.pull(label_image)
        
        return label_image
    except Exception as e:
        print(f"Error processing {image}: {str(e)} on the gpu.")
        return None


def get_ventricle(image):
    try:

        ventricle = gpu_processing(image)
        label_props = regionprops(ventricle)
        largest_label = max(label_props, key=lambda region: region.area)
        print(f"Ventricle label: {largest_label.label}")
        ventricle = np.where(ventricle == largest_label.label, ventricle, 0)
        ventricle = ventricle.astype(np.int64)

        return ventricle
    
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting ventricle.")
        return None

def get_ventricle_wo_injury(image):
    try:
  
        ventricle_wo_injury = np.copy(image)
        ventricle_wo_injury[ventricle_wo_injury == INJURY_LABEL_ID] = 0
        ventricle_wo_injury = gpu_processing(ventricle_wo_injury)
        label_props = regionprops(ventricle_wo_injury)
        largest_label = max(label_props, key=lambda region: region.area)
        ventricle_wo_injury = np.where(ventricle_wo_injury == largest_label.label, ventricle_wo_injury, 0)
        ventricle_wo_injury = ventricle_wo_injury.astype(np.int64)

        return ventricle_wo_injury
    
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting ventricle without injury.")
        return None



def get_injury(ventricle, ventricle_wo_injury):
    try:

        # subtract ventricle_wo_injury from ventricle to get injury
        ventricle = cle.push(ventricle)
        ventricle_wo_injury = cle.push(ventricle_wo_injury)
        injury = cle.binary_subtract(ventricle, ventricle_wo_injury)
        injury = cle.pull(injury)
        injury = injury.astype(np.int64)

        return injury
    except Exception as e:
        print(f"Error processing {image}: {str(e)}  while getting injury.")
        return None



def get_epicardium(ventricle):
    try:

        ventricle = cle.push(ventricle)
        not_ventricle = cle.binary_not(ventricle)
        not_ventricle_dilated = cle.dilate_labels(not_ventricle, None, EPICARDIUM_DIAMETER_PX)
        epicardium = cle.binary_and(not_ventricle_dilated, ventricle)
        epicardium = cle.dilate_labels(epicardium, None, EPICARDIUM_ADDITIONAL_DIAMETER_PX)
        epicardium = cle.binary_subtract(epicardium, ventricle)
        epicardium = cle.connected_components_labeling_box(epicardium)
        epicardium = cle.pull(epicardium)
        label_props = regionprops(epicardium)
        largest_label = max(label_props, key=lambda region: region.area)
        epicardium = np.where(epicardium == largest_label.label, epicardium, 0)
        epicardium = epicardium.astype(np.int64)

        return epicardium
    
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting epicardium.")
        return None


def get_border_zone(injury, ventricle_wo_injury):
    try:

        injury = cle.push(injury)
        ventricle_wo_injury = cle.push(ventricle_wo_injury)
        injury_dilated = cle.dilate_labels(injury, None, BORDER_ZONE_DIAMETER_PX)
        border_zone = cle.binary_and(injury_dilated, ventricle_wo_injury)
        border_zone = cle.pull(border_zone)
        border_zone = border_zone.astype(np.int64)

        return border_zone
    
    except Exception as e:
        print(f"Error processing {image}: {str(e)} while getting border zone.")
        return None



image_folder = os.path.join(args.input)
for filename in os.listdir(image_folder):
    if not filename.endswith("_labels.tif"):
        continue
    print(f"Processing image: {filename}")
    
    image = imread(os.path.join(image_folder, filename))
    
    ventricle = get_ventricle(image)
    ventricle_wo_injury = get_ventricle_wo_injury(image)
    injury = get_injury(ventricle, ventricle_wo_injury)
    epicardium = get_epicardium(ventricle)
    border_zone = get_border_zone(injury, ventricle_wo_injury)
    if ventricle is not None:
        tf.imwrite(os.path.join(image_folder, filename.replace("_labels.tif", "_ventricle.tif")), ventricle,compression='zlib')
    if injury is not None:
        tf.imwrite(os.path.join(image_folder, filename.replace("_labels.tif", "_injury.tif")), injury,compression='zlib')
    if ventricle_wo_injury is not None:
        tf.imwrite(os.path.join(image_folder, filename.replace("_labels.tif", "_ventricle_wo_injury.tif")), ventricle_wo_injury,compression='zlib')
    if epicardium is not None:
        tf.imwrite(os.path.join(image_folder, filename.replace("_labels.tif", "_epicardium.tif")), epicardium,compression='zlib')
    if border_zone is not None:
        tf.imwrite(os.path.join(image_folder, filename.replace("_labels.tif", "_border_zone.tif")), border_zone,compression='zlib')
