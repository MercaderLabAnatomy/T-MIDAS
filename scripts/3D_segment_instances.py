# this script takes two folders with TIF files as input: one with nuclei channel and one with nuclei channel
# it requires to be run in the napari-apoc conda environment
# https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification



import os
import numpy as np
import argparse
import pyclesperanto_prototype as cle
from napari.utils import io as napari_io
from skimage.io import imread, imsave
import pandas as pd
import apoc
import napari_segment_blobs_and_things_with_membranes as nsbatwm 

# image_folder = "/mnt/disk1/Marco/Marwa/TIFs/"

def load_image(filepath, nuclei_channel):
    try:
        img = imread(filepath)  
       # remove first axis 
        img = img[0,:,:,:,:]
        img.shape
        nuclei_channel_img = img[:,:,:,nuclei_channel]

        return nuclei_channel_img
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# image = load_image(filepath, 1) 
# image.shape

def get_3D_labels_otsu(image, label_threshold):
    image = cle.push(image)
    blurred = cle.gaussian_blur(image, sigma_x=1, sigma_y=1, sigma_z=1)
    binary = cle.threshold_otsu(blurred)
    labeled = cle.connected_components_labeling_box(binary)
    labeled = cle.exclude_small_labels(labeled, None, label_threshold)
    labeled = cle.pull(labeled)
    return labeled

# get_3D_labels_otsu(image, 500).shape      

def get_3D_labels_threshold(image, label_threshold):
    image = cle.push(image)
    blurred = cle.gaussian_blur(image, sigma_x=1, sigma_y=1, sigma_z=1)
    binary = cle.greater_or_equal_constant(blurred, None, 14.0)
    labeled = cle.connected_components_labeling_box(binary)
    labeled = cle.exclude_small_labels(labeled, None, label_threshold)
    labeled = cle.pull(labeled)
    return labeled




# Define constants
label_threshold = 500.0
SIGMA = 2.0 # dilation of UEPs, cf. https://imagej.nih.gov/ij/docs/menus/process.html#watershed

# Argument parsing
parser = argparse.ArgumentParser(description='Get 3D nuclei segmentation')
parser.add_argument('--image_folder', type=str, help='Path to folder containing TIF files')
parser.add_argument('--nuclei_channel', type=int, help='Channel to use as nuclei channel')
args = parser.parse_args()


nuclei_channel = args.nuclei_channel

# nuclei_channel = 1

# Prompt user for input folders
image_folder = args.image_folder


for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".tif") or image_filename.endswith(".tiff"):
        filepath = os.path.join(image_folder, image_filename)
        print(f"Segmenting {filepath}")
        nuclei_image = load_image(filepath,nuclei_channel)
        print(nuclei_image.shape)
        nuclei_labels = get_3D_labels_otsu(nuclei_image, label_threshold)
        print(nuclei_labels.shape)
        napari_io.imsave(os.path.join(image_folder,image_filename.split(".")[0] + "_nuclei_intensities.tif"), nuclei_image)
        napari_io.imsave(os.path.join(image_folder,image_filename.split(".")[0] + "_nuclei_labels.tif"), nuclei_labels)
 
