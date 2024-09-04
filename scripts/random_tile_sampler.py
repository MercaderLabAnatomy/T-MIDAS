import numpy as np
import random
import argparse
import tifffile as tf
import os
from tqdm import tqdm

"""
Description: This script reads a tiff image, samples random tiles from the image, and saves the tiles as tiff files.

The output tiff files are saved in a folder named "random_tiles" in the same directory as the input tiff images.

The random sampling is based on the tile diagonal and the percentage of random tiles to be picked from the entire image.

The user can also specify a random seed for reproducibility.

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Sample random tiles from a tiff image.')
    parser.add_argument('--input', type=str, help='Folder containing the .tif images.')
    parser.add_argument('--tile_diagonal', type=int, help='Enter the tile diagonal in pixels.')
    parser.add_argument('--percentage', type=int, help='Enter the percentage of random tiles to be picked from the entire image (20-100).')
    parser.add_argument('--random_seed', type=int, help='Enter a random seed for reproducibility (integer): ')
    return parser.parse_args()

args = parse_args()

def load_tiff_image(path):
    return tf.imread(path)

def is_multichannel(image):
    return len(image.shape) > 2

# the following function creates a grid based on image xy shape and tile diagonal and then randomly samples 20% of the available tiles
def sample_tiles_random(image, tile_diagonal, subset_percentage, random_seed):
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
    random.seed(random_seed)
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

def save_tiles(tiles, path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, tile in enumerate(tiles, start=1):  # Start enumeration at 1
        filename = os.path.basename(path)
        filename = filename.split('.')[0] + '_tile_' + str(i).zfill(2) + '.tif'
        save_path = os.path.join(output_dir, filename)
        tf.imwrite(save_path, tile)



def process_image(path, tile_diagonal, output_dir, subset_percentage, random_seed):
    image = load_tiff_image(path)
    tiles = sample_tiles_random(image, tile_diagonal, subset_percentage, random_seed)
    save_tiles(tiles, path, output_dir)

def main():

    output_dir = os.path.join(args.input, 'random_tiles')
    for filename in tqdm(os.listdir(args.input), total=len(os.listdir(args.input)), desc="Processing images"):
        if filename.endswith(".tif"):
            path = os.path.join(args.input, filename)
            process_image(path, args.tile_diagonal, output_dir, args.percentage, args.random_seed)
        else:
            continue

if __name__ == "__main__":
    main()
