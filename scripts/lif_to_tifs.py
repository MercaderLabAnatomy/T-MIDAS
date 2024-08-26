import os
import argparse
import numpy as np
import cupy as cp
import tifffile as tf
from readlif.reader import LifFile
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Process a lif file.')
    parser.add_argument('--input', type=str, help='path to the lif file')
    return parser.parse_args()

args = parse_args()

input_folder = args.input

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
            for t in range(scene.nt):    
                frame = scene.get_frame(t=t,c=i,z=j)
                # convert frame to numpy array
                frame = cp.array(frame)
                frames.append(frame)
        frames = cp.array(frames)
        array_list.append(frames)

    multichannel_stack = cp.array(array_list)
    if len(multichannel_stack.shape) < 5:
         multichannel_stack = cp.expand_dims(multichannel_stack, axis=0)
    else:
        pass    


    return multichannel_stack


def create_metadata(scene):
    scale_x = scene.info['scale_n'][1]
    scale_y = scene.info['scale_n'][2]
    
    # check if scale_z is defined in the dictionary
    if 3 in scene.info['scale_n']:
        scale_z = scene.info['scale_n'][3]
        z_res = 1 / scale_z
    else:
        z_res = None  # Set to None for 2D images
    
    # get resolution in um
    x_res = 1 / scale_x 
    y_res = 1 / scale_y 
    resolution = (x_res, y_res)
    metadata = {'spacing': z_res, 'unit': 'um'}
    return resolution, metadata


def save_image(image, res_meta, path):
    resolution, metadata = res_meta
    if metadata['spacing'] is None:
        # For 2D images, don't include spacing in metadata
        metadata.pop('spacing')
    tf.imwrite(path, image, resolution=resolution, metadata=metadata, imagej=True, compression='zlib')


def process_scene(scene,path):
    multichannel_stack = cp.asnumpy(scene_to_stack(scene))
    
    # create metadata
    res_meta = create_metadata(scene)
    position = scene.info['name']
    # replace "/" with "_" in position
    position = position.replace("/","_")
    # save the multichannel stack
    save_image(multichannel_stack,res_meta,path.split(".")[0] +"_{scene}.tif".format(scene=position))
    print("Processed {scene}".format(scene=position))

for lif_file in tqdm(lif_files, total = len(lif_files), desc="Processing lif files"):
    
    if lif_file:
        
        path = os.path.join(input_folder, lif_file)
        # read the lif file
        lif = LifFile(path)
        # index all scenes in the lif file
        img_list = [i for i in lif.get_iter_image()]

        for scene in img_list:
            process_scene(scene,path)    

    else:
        print("No lif file found in the input folder.")
        exit()











