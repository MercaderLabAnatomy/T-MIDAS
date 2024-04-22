"""
Script to extract regions of interest (ROIs) from NDPI files and save them as TIF files.

Requires folder containing single channel NDPI files with an NDPIS file with the following filename pattern:

filename.ndpis
filename-DAPI.ndpi
...

ROIs are extracted by finding the contours in a binary image and are then filtered by size (keep big hearts, skip the rest). 
The binary image is created using blurring and otsu thresholding. 
The user is prompted to select the channel from which the ROIs should be extracted. 
The same ROIs are then used to crop the images from the other channels. 

Installation and usage instructions can be found at the bottom of this script.

This is not the fastest script (single CPU core), but it does the job on its own.
It works well for NDPI files of around 200-300MB, requiring about 20GB of RAM.
It takes between 1-5 minutes per slide with 0.23-0.46um/pixel resolution. 
The user is prompted to select either resolution level.

"""

import openslide
import os

from PIL import ImageFilter
from PIL import Image

import numpy as np
import argparse
import tifffile as tf
import pyclesperanto_prototype as cle
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
import napari_segment_blobs_and_things_with_membranes as nsbatwm  # version 0.3.7
from skimage.measure import regionprops
# ignore warnings
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Extract ROIs from NDPI files and save them as TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the NDPI(s) files.')
    parser.add_argument('--cropping_template_channel_name', type=str, help='Enter the channel name that represents the cropping template (hearts = FITC).')
    # parser.add_argument('--level', type=int, help='Enter the resolution level of the NDPI image (0 = highest resolution, 1 = second highest resolution).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input

# ask for the channel that contains the cropping template
CROPPING_TEMPLATE_CHANNEL_NAME = args.cropping_template_channel_name


# adjust THRESHOLD_SIZE to resolution level of ndpi image



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

def get_rois(template_ndpi_file):

    slide = openslide.OpenSlide(os.path.join(input_folder, template_ndpi_file))
    scaling_factor = 100
    slide_dims_downscaled = (slide.dimensions[0] / scaling_factor, slide.dimensions[1] / scaling_factor)
    thumbnail = slide.get_thumbnail(slide_dims_downscaled)
    thumbnail = thumbnail.convert('L')
    labeled_thumbnail = nsbatwm.gauss_otsu_labeling(thumbnail, 10.0)
    props = regionprops(labeled_thumbnail)
    rois = []
    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - 10)
        minc = max(0, minc - 10)
        maxr = min(thumbnail.height, maxr + 10)
        maxc = min(thumbnail.width, maxc + 10)
        rois.append((minc*scaling_factor, minr*scaling_factor, (maxc-minc)*scaling_factor, (maxr-minr)*scaling_factor))
    return rois


for ndpis_file in ndpis_files:


    ndpi_files = get_ndpi_filenames(os.path.join(input_folder, ndpis_file))
    CROPPING_TEMPLATE_CHANNEL = [ndpi_file for ndpi_file in ndpi_files if CROPPING_TEMPLATE_CHANNEL_NAME in ndpi_file][0]
    rois = get_rois(CROPPING_TEMPLATE_CHANNEL)
    number_of_rois = len(rois)

    for ndpi_file in ndpi_files:
        if ndpi_file.endswith(".ndpi"):

            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
            slide = openslide.OpenSlide(os.path.join(input_folder, ndpi_file))
            for i, roi in enumerate(rois):
                x, y, w, h = roi

                cropped_image = slide.read_region((x, y), 0, (w, h))
                cropped_image_dimensions = cropped_image.size
                print("ROI %d of %d with dimensions %s saved as %s" % (i+1, number_of_rois, cropped_image_dimensions, output_filename + "_roi_0" + str(i+1) + ".tif"))
                cropped_image = cropped_image.convert('L')
                cropped_image.save(output_filename + "_roi_0" + str(i+1) + ".tif")
                # tf.imwrite(output_filename + "_roi_0" + str(i+1) + "_label.tif", label_image, compression='zlib')

"""

1) Installation (Ubuntu)

1.1) Install OpenSlide

sudo apt install openslide-tools

1.2) Install fast package manager Mamba

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

1.3) Other operating systems (Windows, Mac OS)

https://openslide.org/api/python/#installing
https://github.com/conda-forge/miniforge 


2) Create Mamba environment and install dependencies

mamba create -n openslide-env openslide-python
mamba activate openslide-env
pip install opencv-python


3) Usage

In CLI, type

mamba activate openslide-env
python NDPI2TIF.py

"""
