
import random
import os
#import cv2
import numpy as np
import argparse
import tifffile as tf # imread gives dim order CXY !
import pyclesperanto_prototype as cle
from skimage.measure import label
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pyclesperanto_prototype as cle
from PIL import Image
#import torch
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import napari_simpleitk_image_processing as nsitk  # version 0.4.5
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from skimage.io import imread, imsave # gives dim order XYC !



def parse_arguments():
    parser = argparse.ArgumentParser(description='ndpi to csv.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the multichannel .tif files.')
    parser.add_argument('--channels', nargs='+', help='Enter the names of the channels in the order they appear in the .tif files.\n Example: DAPI GFP RFP')
    parser.add_argument('--tile_diagonal', type=int, help='Enter the tile diagonal in pixels.')
    parser.add_argument('--random_seed', type=int, help='Enter the random seed for reproducibility.')
    parser.add_argument('--percentage', type=int, help='Enter the percentage of random tiles to be picked from the entire image (20-100).')
    
    return parser.parse_args()


# Check if CUDA is available
#print("CUDA is available:", torch.cuda.is_available())

# Set up mask generator
# model_type = "vit_h"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint='/opt/ML_models/sam_vit_h_4b8939.pth')#args.checkpoint)
# sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam)



def make_output_dirs(input_folder, channel_names):
    output_folder = os.path.join(input_folder, "random_tiles")
    os.makedirs(output_folder, exist_ok=True)  # Create the 'random_tiles' directory

    for channel_name in channel_names:
        channel_path = os.path.join(output_folder, channel_name)
        os.makedirs(channel_path, exist_ok=True)  # Create subdirectories for each channel

    return output_folder

def is_multichannel(image):
    return len(image.shape) > 2



# the following function creates a grid based on image xy shape and tile diagonal and then randomly samples 20% of the available tiles
# the following function creates a grid based on image xy shape and tile diagonal and then randomly samples 20% of the available tiles
def sample_tiles_random(image, tile_diagonal, subset_percentage):
    tiles = []
    
    if is_multichannel(image) and (image.shape[0] < 5): # to account for both cxy and xyc, where c < 5 (less than 5 colors)

        height, width = image.shape[1], image.shape[2]
    else:
        height, width = image.shape[0], image.shape[1]
            
    # print("image shape: ("+str(height)+","+str(width)+")\n")
    tile_size = int(np.sqrt(2) * tile_diagonal)  # Calculate the tile size
    
    step_h = int(tile_size)  # Set step sizes based on the tile size
    step_w = int(tile_size)
    
    possible_positions = []  # Store all possible tile positions
    
    for i in range(0, height - tile_size + 1, step_h):
        for j in range(0, width - tile_size + 1, step_w):
            possible_positions.append((i, j))  # Collect all possible tile positions
    
    num_subset_tiles = int(len(possible_positions) * (subset_percentage / 100))  # Calculate number of tiles for subset
    random.seed(args.random_seed)
    selected_positions = random.sample(possible_positions, num_subset_tiles)  # Randomly select non-overlapping positions
    
    if is_multichannel(image) and (image.shape[0] < 5):
        for pos in selected_positions:
            i, j = pos
            tile = image[:,i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
    else:
        for pos in selected_positions:
            i, j = pos
            tile = image[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
    
    return tiles

def split_channels(image):
    n_channels = image.shape[2] if len(image.shape) > 2 else 1 # Check if image is multichannel, expects 3rd dimension to be channels
    channels = []
    if n_channels > 1:
        for i in range(n_channels):
            channel = image[:, :,i]
            channels.append(channel)
    else:
        channels.append(image)
    return channels

def save_channels(tile_channels, save_path, channel_names):

    for i, tile_channel in enumerate(tile_channels):  
        channel_name = channel_names[i]
        filename = os.path.basename(save_path)
        output_dir = os.path.dirname(save_path)
        filename = 'C'+str(i+1)+'-'+filename.split('.')[0] + '.tif'
        tc_save_path = os.path.join(output_dir, channel_name, filename)
        imsave(tc_save_path, tile_channel)

def get_tif_files(input_folder):
    tif_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.endswith('.tif')]
    return tif_files

def process_dapi_image(array):
    try:
        image = Image.fromarray(array, 'L')
        image = cle.push(image)
        image_gb = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
        image_to = cle.threshold_otsu(image_gb)
        image_l = cle.connected_components_labeling_box(image_to)
        image_S = nsbatwm.split_touching_objects(image_l, 9.0)
        image_labeled = cle.connected_components_labeling_box(image_S)
        image_labeled = cle.pull(image_labeled)
        return image_labeled
    except Exception as e:
        print(f"Error processing {array}: {str(e)}")
        return None

def process_fitc_image(array, mask_generator):
    try:
        image = Image.fromarray(array, 'L')
        masks = mask_generator.generate(image)
        return masks
    except Exception as e:
        print(f"Error processing {array}: {str(e)}")
        return None

def process_tritc_image(array):
    try:
        image = Image.fromarray(array, 'L')
        image = cle.push(image) 
        image_gb = cle.gaussian_blur(image, None, 1.0, 1.0, 0.0)
        image_thb = cle.top_hat_box(image_gb, None, 10.0, 10.0, 0.0)
        image_t = nsitk.threshold_renyi_entropy(image_thb)
        image_cclb = cle.connected_components_labeling_box(image_t)
        result_esl = cle.exclude_labels_outside_size_range(image_cclb, None, 200.0, 1200.0)
        result_esl = cle.pull(result_esl)
        return result_esl
    except Exception as e:
        print(f"Error processing {array}: {str(e)}")
        return None

def process_fitc_image_classical(array):
    image = Image.fromarray(array, 'L')
    image = cle.push(image)
    labels = cle.gauss_otsu_labeling(image, None, 20.0)
    labels = cle.pull(labels)
    return labels


def process_multichannel_tifs(input_folder, tile_diagonal, channel_names, subset_percentage):
    """
    this is the main function that processes the multichannel tif files 
    by sampling random tiles and saving them, and then processing the tiles
    """
    tif_files = get_tif_files(input_folder)
    tile_dir = make_output_dirs(input_folder, channel_names)
    for tif_file in tif_files:
        tiff_image = imread(os.path.join(input_folder, tif_file))
        tiles = sample_tiles_random(tiff_image, tile_diagonal, subset_percentage)
        print("Number of tiles:", len(tiles))
        for i, tile in enumerate(tiles, start=1):
            print(i,tile.shape)
            filename = os.path.basename(tif_file)
            filename = filename.split('.')[0] + '_tile_' + str(i).zfill(2) + '.tif'
            save_path = os.path.join(tile_dir, filename)
            print("save_path:", save_path)
            imsave(save_path, tile)

            tile_channels = split_channels(tile)
            save_channels(tile_channels, save_path, channel_names)
            for i,channel_name in enumerate(channel_names):
                if channel_name == "DAPI":
                    label_image = process_dapi_image(tile_channels[2])
                elif channel_name == "FITC":
                    label_image = process_fitc_image_classical(tile_channels[1])
                elif channel_name == "TRITC":
                    label_image = process_tritc_image(tile_channels[0])
                elif channel_name == "CY5":
                    label_image = process_fitc_image_classical(tile_channels[3])
                else:
                    print("Channel name not recognized.")
                if label_image is not None:
                    label_filename = 'C'+str(i+1)+'-'+filename.split('.')[0] + '_labels.tif'
                    imsave(os.path.join(tile_dir, channel_name, label_filename), label_image)#, compression='zlib')

def main():
    args = parse_arguments()
    if args.channels:
        print('Channels provided:', args.channels)
        channel_list = [c.upper() for c in args.channels]
    else:
        print('No channels provided.')
    process_multichannel_tifs(args.input, args.tile_diagonal, channel_list, args.percentage)

if __name__ == "__main__":
    main()


