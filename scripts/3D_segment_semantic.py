# this script takes two folders with TIF files as input: one with nuclei channel and one with tissue channel
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


# image_folder = "/mnt/disk1/Marco/Marwa/"
# tissue_channel = 2

def load_image(filepath, tissue_channel):
    try:
        img = imread(filepath)  
       # remove first axis 
        img = img[0,:,:,:,:]
        tissue_channel_img = img[:,:,:,tissue_channel]

        return tissue_channel_img
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

cl_filename = "/opt/scripts/ObjectSegmenter_weaker_tissues.cl"

def get_3D_labels_RandomForestClassifier(image, label_threshold):
    image = cle.push(image)
    blurred = cle.gaussian_blur(image, sigma_x=1, sigma_y=1, sigma_z=1)
    clf = apoc.ObjectSegmenter(opencl_filename=cl_filename)
    binary = clf.predict(image=blurred)
    labeled = cle.erode_labels(binary, None, 3.0, True)
    labeled = cle.dilate_labels(labeled, None, 2.0)
    labeled = cle.connected_components_labeling_box(labeled)
    labeled = cle.exclude_small_labels(labeled, None, label_threshold)
    labeled = cle.pull(labeled)
    return labeled



# Define constants
label_threshold = 100000.0

SIGMA = 2.0 # dilation of UEPs, cf. https://imagej.nih.gov/ij/docs/menus/process.html#watershed

# Argument parsing
parser = argparse.ArgumentParser(description='Get 3D tissue segmentation')
parser.add_argument('--image_folder', type=str, help='Path to folder containing TIF files')
parser.add_argument('--tissue_channel', type=int, help='Channel to use as tissue channel')
args = parser.parse_args()



# Prompt user for input folders
image_folder = args.image_folder


for image_filename in os.listdir(image_folder):
    if image_filename.endswith(".tif") or image_filename.endswith(".tiff"):
        filepath = os.path.join(image_folder, image_filename)
        print(f"Segmenting {filepath}")
        tissue_image = load_image(filepath,tissue_channel)
        print(tissue_image.shape)
        tissue_labels = get_3D_labels_RandomForestClassifier(tissue_image, label_threshold)
        print(tissue_labels.shape)
        napari_io.imsave(os.path.join(image_folder,image_filename.split(".")[0] + "_tissue.tif"), tissue_labels)
                


