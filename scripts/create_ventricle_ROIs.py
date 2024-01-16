#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
from skimage.measure import regionprops



# Argument Parsing
parser = argparse.ArgumentParser(description="Input: Folder with label images containing masks of intact ventricle and injury regions.")
parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
args = parser.parse_args()

EPICARDIUM_DIAMETER_UM = 15.0
EPICARDIUM_ADDITIONAL_DIAMETER_UM = 10.0
PIXEL_RESOLUTION = 0.23 # Slidescanner Hamamatsu S60 at 40x
EPICARDIUM_DIAMETER_PX = EPICARDIUM_DIAMETER_UM / PIXEL_RESOLUTION
EPICARDIUM_ADDITIONAL_DIAMETER_PX = EPICARDIUM_ADDITIONAL_DIAMETER_UM / PIXEL_RESOLUTION
INTACT_VENTRICLE_REGION_LABEL_ID = 1
INJURY_LABEL_ID = 2
BORDER_ZONE_DIAMETER_UM = 100.0
BORDER_ZONE_DIAMETER_PX = BORDER_ZONE_DIAMETER_UM / PIXEL_RESOLUTION
SMALL_LABELS_THRESHOLD = 10000.0 # square pixels

# Define utility functions


def get_ventricle(image_path):
    try:

        image = imread(image_path)        
        ventricle = cle.merge_touching_labels(image)
        ventricle = nsitk.binary_fill_holes(np.array(ventricle))
        ventricle = cle.connected_components_labeling_box(ventricle)
        ventricle = cle.exclude_small_labels(ventricle, None, SMALL_LABELS_THRESHOLD)
        ventricle = np.array(ventricle, dtype=np.uint64)

        return ventricle
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)} while getting ventricle.")
        return None



# image_path = "MF20/wt_7dpi_5_2 - 3 - CY5_8bit_labels.tif"


# import napari
# if 'viewer' not in globals():
#     viewer = napari.Viewer()

# ventricle = get_ventricle(image_path)
# viewer.add_labels(ventricle)
# tf.imwrite(image_path+ "_ventricle.tif", ventricle,compression='zlib')


def get_ventricle_wo_injury(image_path):
    try:

        image = imread(image_path)     
        ventricle_wo_injury = np.copy(image)
        ventricle_wo_injury[ventricle_wo_injury == INJURY_LABEL_ID] = 0
        ventricle_wo_injury = nsitk.binary_fill_holes(ventricle_wo_injury)
        ventricle_wo_injury = cle.connected_components_labeling_box(ventricle_wo_injury)
        label_props = regionprops(ventricle_wo_injury)
        largest_label = max(label_props, key=lambda region: region.area)
        ventricle_wo_injury = np.where(ventricle_wo_injury == largest_label, ventricle_wo_injury, 0)
        ventricle_wo_injury = np.array(ventricle_wo_injury, dtype=np.uint64)

        return ventricle_wo_injury
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)} while getting ventricle without injury.")
        return None


def get_injury(image_path):
    try:

        image = imread(image_path)
        injury = np.copy(image)
        injury[injury == INTACT_VENTRICLE_REGION_LABEL_ID] = 0
        injury = nsitk.binary_fill_holes(injury)
        injury = cle.connected_components_labeling_box(injury)
        label_props = regionprops(injury)
        largest_label = max(label_props, key=lambda region: region.area)
        injury = np.where(injury == largest_label, injury, 0)
        injury = np.array(injury, dtype=np.uint64)

        return injury
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}  while getting injury.")
        return None

# injury = get_injury(image_path)

# tf.imwrite(image_path+ "_injury.tif", injury)

def get_epicardium(image_path):
    try:

        image = imread(image_path)
        
        ventricle = cle.merge_touching_labels(image)
        ventricle = cle.exclude_small_labels(ventricle, None, SMALL_LABELS_THRESHOLD)
        ventricle = nsitk.binary_fill_holes(np.array(ventricle))
        #viewer.add_labels(ventricle)
        not_ventricle = nsbatwm.binary_invert(ventricle)
        not_ventricle_expanded = nsbatwm.expand_labels(not_ventricle, EPICARDIUM_DIAMETER_PX)
        #viewer.add_labels(not_ventricle)
        #viewer.add_labels(not_ventricle_expanded)
        epicardium = cle.binary_and(not_ventricle_expanded, ventricle)
        ventricle = cle.binary_subtract(ventricle, epicardium)
        epicardium = nsbatwm.expand_labels(epicardium, EPICARDIUM_ADDITIONAL_DIAMETER_PX)
        epicardium = cle.binary_subtract(epicardium, ventricle)
        epicardium = cle.connected_components_labeling_box(epicardium)
        label_props = regionprops(epicardium)
        largest_label = max(label_props, key=lambda region: region.area)
        epicardium = np.where(epicardium == largest_label, epicardium, 0)
        #viewer.add_labels(epicardium)
        # convert epicardium to ndarray
        epicardium = np.array(epicardium, dtype=np.uint64)

        return epicardium
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)} while getting epicardium.")
        return None

# epicardium = get_epicardium(image_path)

# tf.imwrite(image_path+ "_epicardium.tif", epicardium,compression='zlib')

# imread(image_path+ "_epicardium.tif")

def get_border_zone(image_path):
    try:

        image = imread(image_path)      
        ventricle = cle.merge_touching_labels(image)
        ventricle = cle.exclude_small_labels(ventricle, None, SMALL_LABELS_THRESHOLD)
        injury = np.copy(image)
        injury[injury == INTACT_VENTRICLE_REGION_LABEL_ID] = 0
        injury = nsitk.binary_fill_holes(injury)
        injury = cle.exclude_small_labels(injury, None, SMALL_LABELS_THRESHOLD)
        injury_expanded = nsbatwm.expand_labels(injury, BORDER_ZONE_DIAMETER_PX)
        ventricle_wo_injury = np.copy(image)
        ventricle_wo_injury[ventricle_wo_injury == INJURY_LABEL_ID] = 0
        ventricle_wo_injury = nsitk.binary_fill_holes(ventricle_wo_injury)
        ventricle_wo_injury = cle.connected_components_labeling_box(ventricle_wo_injury)
        ventricle_wo_injury = cle.exclude_small_labels(ventricle_wo_injury, None, SMALL_LABELS_THRESHOLD)
        border_zone = cle.binary_and(ventricle_wo_injury, injury_expanded)
        border_zone = cle.connected_components_labeling_box(border_zone)
        label_props = regionprops(border_zone)
        largest_label = max(label_props, key=lambda region: region.area)
        border_zone = np.where(border_zone == largest_label, border_zone, 0)
        border_zone = np.array(border_zone, dtype=np.uint64)

        return border_zone
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)} while getting border zone.")
        return None

# image_folder = "/mnt/01D9F776C2451850/Carla/labels"
# Process images
image_folder = os.path.join(args.input)
for filename in os.listdir(image_folder):
    if not filename.endswith("_labels.tif"):
        continue
    print(f"Processing image: {filename}")
    ventricle = get_ventricle(os.path.join(image_folder, filename))
    injury = get_injury(os.path.join(image_folder, filename))
    ventricle_wo_injury = get_ventricle_wo_injury(os.path.join(image_folder, filename))
    epicardium = get_epicardium(os.path.join(image_folder, filename))
    border_zone = get_border_zone(os.path.join(image_folder, filename))
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
