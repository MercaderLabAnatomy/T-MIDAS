#!/usr/bin/env python
# coding: utf-8

import os
import csv
import argparse
import numpy as np
from skimage.io import imread
import tifffile as tf
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import re
from skimage.measure import regionprops

# Argument Parsing
parser = argparse.ArgumentParser(description="Input: Folder with all label images (ROIs and instance segmentations).")
parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
args = parser.parse_args()


# def get_area(ROI):
#     area = cle.sum_of_all_pixels(ROI)
#     return area

# def get_circularity(ROI):
#     region = regionprops(ROI)
#     circularity = 4.0 * np.pi * region.area / (region.perimeter ** 2.0) 
#     return circularity

# def get_AR(ROI):
#     region = regionprops(ROI)
#     AR = region.major_axis_length / region.minor_axis_length
#     return AR

def counter(ROI,instances):
    instances_inside_ROI = cle.binary_and(ROI, instances)
    instances_inside_ROI = cle.connected_components_labeling_box(instances_inside_ROI)
    count = int(cle.maximum_of_all_pixels(instances_inside_ROI))
    return count


# define function that counts instances per ROI
def ROI2CSV(image_path):

    # load label images
    ventricle_wo_injury = imread(image_path.replace(".tif", "_ventricle_wo_injury.tif"))
    injury = imread(image_path.replace(".tif", "_injury.tif"))
    epicardium = imread(image_path.replace(".tif", "_epicardium.tif"))
    border_zone = imread(image_path.replace(".tif", "_border_zone.tif"))
    instances = imread(image_path.replace(".tif", "_labels.tif").replace("FITC", "CY5"))

    # create CSV file
    filename = image_path.replace(".tif", ".csv")
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["ROI", "instances"])#,"ROI_area (pixels squared)","circularity","aspect_ratio"])
        writer.writerow(["ventricle_wo_injury", counter(ventricle_wo_injury,instances)])#, get_area(ventricle_wo_injury), get_circularity(ventricle_wo_injury), get_AR(ventricle_wo_injury)])
        writer.writerow(["injury", counter(injury,instances)])#, get_area(injury), get_circularity(injury), get_AR(injury)])
        writer.writerow(["epicardium", counter(epicardium,instances)])#, get_area(epicardium), get_circularity(epicardium), get_AR(epicardium)])
        writer.writerow(["border_zone", counter(border_zone,instances)])#, get_area(border_zone), get_circularity(border_zone), get_AR(border_zone)])



# create list of filenames
filenames = os.listdir(args.input)
# filter out files that don't end with .tif


filenames = [s for s in filenames if "CY5" not in s]
new_filenames = []
print(filenames)
for filename in filenames:
    if filename.endswith(".tif"):
        match = re.search(r'.+_roi_\d{2,3}', filename)
        if match:
            index = match.group(0)
            new_filename = filename[:match.end()] + ".tif"
            new_filenames.append(new_filename)

# remove duplicates
new_filenames = list(dict.fromkeys(new_filenames))
new_filenames[0]
for filename in new_filenames:

    if not filename.endswith(".tif"):
        continue
    print(f"Processing image: {filename}")
    ROI2CSV(os.path.join(args.input, filename))
