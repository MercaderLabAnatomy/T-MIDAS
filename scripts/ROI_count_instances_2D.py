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
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="Input: Folder with all label images (ROIs and instance segmentations).")
parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
parser.add_argument("--input", type=str, required=True, help="Path to input label images.")
args = parser.parse_args()


def get_area(ROI):
    area = cle.sum_of_all_pixels(ROI)
    area_mm2 = area * (args.pixel_resolution / 1000)**2
    return area_mm2

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
def ROI2CSV(original_filepath, instance_filepath, ventricle_wo_injury_filepath, injury_filepath, epicardium_filepath, border_zone_filepath):

    # load label images
    instances = imread(instance_filepath)
    ventricle_wo_injury = imread(ventricle_wo_injury_filepath)
    injury = imread(injury_filepath)
    epicardium = imread(epicardium_filepath)
    border_zone = imread(border_zone_filepath)


    # create CSV file
    filename = original_filepath.replace(".tif", ".csv")
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["ROI", "instances","ROI area (sq. mm)"])#,"circularity","aspect_ratio"])
        writer.writerow(["ventricle_wo_injury", counter(ventricle_wo_injury,instances), get_area(ventricle_wo_injury)])#, get_circularity(ventricle_wo_injury), get_AR(ventricle_wo_injury)])
        writer.writerow(["injury", counter(injury,instances), get_area(injury)])#, get_circularity(injury), get_AR(injury)])
        writer.writerow(["epicardium", counter(epicardium,instances), get_area(epicardium)])#, get_circularity(epicardium), get_AR(epicardium)])
        writer.writerow(["border_zone", counter(border_zone,instances), get_area(border_zone)])#, get_circularity(border_zone), get_AR(border_zone)])



# create list of filenames
filenames = os.listdir(args.input)
# filter out files that don't end with .tif


filenames = [s for s in filenames if "FITC" not in s]
new_filenames = []

for filename in filenames:
    if filename.endswith(".tif"):
        match = re.search(r'.+_roi_\d{2,3}', filename)
        if match:
            index = match.group(0)
            new_filename = filename[:match.end()] + ".tif"
            new_filenames.append(new_filename)

# remove duplicates
original_filenames = list(set(new_filenames))


instance_filenames = [f.replace(".tif", "_labels.tif") for f in original_filenames]
ventricle_wo_injury_filenames = [f.replace(".tif", "_ventricle_wo_injury.tif").replace("CY5", "FITC") for f in original_filenames]
injury_filenames = [f.replace(".tif", "_injury.tif").replace("CY5", "FITC") for f in original_filenames]
epicardium_filenames = [f.replace(".tif", "_epicardium.tif").replace("CY5", "FITC") for f in original_filenames]
border_zone_filenames = [f.replace(".tif", "_border_zone.tif").replace("CY5", "FITC") for f in original_filenames]


original_filepaths = [os.path.join(args.input, filename) for filename in original_filenames]
instance_filepaths = [os.path.join(args.input, filename) for filename in instance_filenames]
ventricle_wo_injury_filepaths = [os.path.join(args.input, filename) for filename in ventricle_wo_injury_filenames]
injury_filepaths = [os.path.join(args.input, filename) for filename in injury_filenames]
epicardium_filepaths = [os.path.join(args.input, filename) for filename in epicardium_filenames]
border_zone_filepaths = [os.path.join(args.input, filename) for filename in border_zone_filenames]


# iterate over length of list
for i in range(len(original_filepaths)):
    print(f"Processing image: {original_filepaths[i]}")
    ROI2CSV(original_filepaths[i], instance_filepaths[i], ventricle_wo_injury_filepaths[i], injury_filepaths[i], epicardium_filepaths[i], border_zone_filepaths[i])



# for filename in new_filenames:

#     if not filename.endswith(".tif"):
#         continue
#     print(f"Processing image: {filename}")
#     ROI2CSV(os.path.join(args.input, filename))
