from readlif.reader import LifFile
import numpy as np
import cv2
import os
import pyclesperanto_prototype as cle
import tifffile as tf
import argparse
from tqdm import tqdm
import javabridge
import bioformats

"""
Description: This script reads a lif file, crops the images to the region of interest, and saves the cropped images as tif files.

This os done by selecting a channel as a template and cropping the images based on the template channel. 

It uses the readlif library to read the lif files and the pyclesperanto library to process the images.

"""


def parse_args():
    parser = argparse.ArgumentParser(description='Process a lif file.')
    parser.add_argument('--input_folder', type=str, help='path to the lif file')
    parser.add_argument('--template_channel', type=int, help='channel to use as template')
    return parser.parse_args()

args = parse_args()

template_channel = args.template_channel
input_folder = args.input_folder



output_dir = input_folder + "/cropped_tif_files"

# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


lif_files = []
for file in os.listdir(input_folder):
    if file.endswith(".lif"):
        lif_files.append(file)





def scene_to_stack(scene):
    n_channels = scene.info['channels']
    array_list = []
    for i in range(n_channels):
        # get the array of the current scene
        frames = []
        for j in range(scene.nz):
        # get the frame
            frame = scene.get_frame(t=0,c=i,z=j)
            # convert frame to numpy array
            frame = np.array(frame)
            frames.append(frame)
        frames = np.array(frames)
        array_list.append(frames)

    multichannel_stack = np.array(array_list)
    return multichannel_stack



def create_metadata(scene):
    scale_x = scene.info['scale_n'][1]
    scale_y = scene.info['scale_n'][2]
    
    # check if scale_z is defined in the dictionary
    if 3 in scene.info['scale_n']:
        scale_z = scene.info['scale_n'][3]
    else:
        scale_z = 1
    
    # get resolution in um
    x_res = 1 / scale_x 
    y_res = 1 / scale_y 
    z_res = 1 / scale_z 
    resolution = (x_res, y_res)
    metadata = {'spacing': z_res, 'unit': 'um'}
    return resolution, metadata



def get_array(scene,template_channel):
    # get the array of the current scene
    frames = []
    n_slices = scene.nz
    for i in range(n_slices):
        # get the frame
        frame = scene.get_frame(t=0,c=template_channel,z=i)
        # convert frame to numpy array
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)
    return frames

def get_template(scene,template_channel):
    n_channels = scene.info['channels']
    template = []
    # if nchannels > 1, select channel to use as template
    if n_channels > 1:
        template = get_array(scene,template_channel)
    else:
        template = get_array(scene,0)
    return template


def get_binary_image(template):

    # push to GPU
    GPU_image = cle.push(template)
    # blur the image to remove noise
    GPU_image = cle.gaussian_blur(GPU_image, sigma_x=25, sigma_y=25, sigma_z=0)
    # binarize using otsu thresholding
    label_image = cle.threshold_otsu(GPU_image)
    # pull back to CPU
    label_image = cle.pull(label_image)
    # set all non-zero values to 255
    label_image[label_image != 0] = 255

    return label_image


def find_largest_rectangle(bounding_rectangles):
    # Calculate the area of each bounding rectangle
    areas = [rect[2] * rect[3] for rect in bounding_rectangles]
    
    # Find the index of the bounding rectangle with the largest area
    largest_index = areas.index(max(areas))
    
    # Get the largest bounding rectangle
    largest_rectangle = bounding_rectangles[largest_index]
    
    return largest_rectangle


def get_bounding_rect(binary_image):

    bounding_rects = []
    for z in range(binary_image.shape[0]):
        # Find contours on the binary slice
        contours, hierarchy = cv2.findContours(binary_image[z, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if there is no contour, do not append the slice
        if len(contours) == 0:
            continue
        else:
            # pick the largest contour
            cnt = max(contours, key=cv2.contourArea)
        # Get bounding rectangle of largest contour
        x, y, w, h = cv2.boundingRect(cnt)

        bounding_rects.append([x, y, w, h])

    bounding_rect = find_largest_rectangle(bounding_rects)

    return bounding_rect


def crop_multichannel_stack(multichannel_stack,bounding_rect):
    
    padding = 100 # add same padding on all sides to prevent cutoffs
    bounding_rect = [bounding_rect[0] - padding, bounding_rect[1] - padding, bounding_rect[2] + 2*padding, bounding_rect[3] + 2*padding]

    # Crop multichannel_stack slicewise along the bounding rectangle
    cropped_frames = [multichannel_stack[:, z, bounding_rect[1]:bounding_rect[1] + bounding_rect[3], 
                                         bounding_rect[0]:bounding_rect[0] + bounding_rect[2]] for z in range(multichannel_stack.shape[1])]
    # stack the frames
    cropped_array = np.stack(cropped_frames, axis=1)
    # return the cropped image
    # add an extra dim for time
    cropped_array = np.expand_dims(cropped_array, axis=2)
    # reorder the axes from CZTXY to TZYXC 
    cropped_array = np.moveaxis(cropped_array, [0, 1, 2, 3, 4], [4, 1, 0, 3, 2])

    return cropped_array



def save_image(image,res_meta,path):
    # save the image stack
    tf.imwrite(path, image,resolution = res_meta[0], metadata=res_meta[1], imagej=True) 

def process_scene(scene,template_channel,path):
    # get the template
    template = get_template(scene,template_channel)
    # get the binary image
    binary_image = get_binary_image(template)
    # get the bounding rectangle
    bounding_rect = get_bounding_rect(binary_image)
    # crop the multichannel stack
    cropped_multichannel_stack = crop_multichannel_stack(scene_to_stack(scene),bounding_rect)
    # create metadata
    res_meta = create_metadata(scene)
    position = scene.info['name']
    # replace "/" with "_" in position
    position = position.replace("/","_")
    # save the cropped multichannel stack
    save_image(cropped_multichannel_stack,res_meta,path.split(".")[0] +"_{scene}_cropped.tiff".format(scene=position))
    # save the cropped binary image
    print("Processed {scene}".format(scene=position))


for lif_file in tqdm(lif_files, total = len(lif_files), desc="Processing lif files"):
    
    if lif_file:
        path = os.path.join(input_folder, lif_file)
        img = LifFile(path)
        img_list = [i for i in img.get_iter_image()]
        for scene in img_list:
            process_scene(scene,template_channel,path)
    else:
        print("Please provide a path to the lif file.")
        exit()
        


