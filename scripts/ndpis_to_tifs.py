
import openslide
import os
import re
from PIL import ImageFilter
import cv2
import numpy as np
import argparse
import tifffile as tf
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
Description: This script reads NDPI files and saves them as TIF files.

The script uses the openslide library to read the NDPI files and the pyclesperanto library to process the images.

The output TIF files are saved in a folder named "tif_files" in the same directory as the input NDPI files.

"""



def parse_args():
    parser = argparse.ArgumentParser(description='Convert NDPI files to TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the NDPI(s) files.')
    parser.add_argument('--level', type=int, help='Enter the resolution level of the NDPI image (0 = highest resolution, 1 = second highest resolution).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input

# ask for the resolution level of the ndpi image
LEVEL = args.level

output_dir = input_folder + "/tif_files"

# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#input_folder = "/media/geffjoldblum/DATA/Romario"

ndpis_files = []
for file in os.listdir(input_folder):
    if file.endswith(".ndpis"):
        ndpis_files.append(file)


def get_ndpi_filenames(ndpis_file):
    ndpi_files = []
    with open(ndpis_file, 'r') as f:
        for line in f:
            if line.endswith('.ndpi\n'):
                # extract substring after "="
                line = line.split("=")[1]
                # save to list            
                ndpi_files.append(line.rstrip('\n'))
        # close file
        f.close()
    return ndpi_files

#get_ndpi_filenames(ndpis_file)



def ndpi_2_tif(ndpi_files):
    ndpi_image = openslide.open_slide(os.path.join(input_folder, ndpi_files)) 
    # Convert the NDPI image to a multichannel tiff image
    tiff_image = ndpi_image.read_region((0, 0), LEVEL, ndpi_image.level_dimensions[LEVEL]).convert('L')
    ndpi_image.close()
    return tiff_image 


for ndpis_file in tqdm(ndpis_files, total = len(ndpis_files), desc="Processing images"):

    ndpi_files = get_ndpi_filenames(os.path.join(input_folder, ndpis_file))

    for ndpi_file in ndpi_files:
        if ndpi_file.endswith(".ndpi"):

            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
            tiff_image = ndpi_2_tif(ndpi_file)
            tiff_image.save(output_filename + ".tif")



