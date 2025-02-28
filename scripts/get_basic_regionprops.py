import os
import argparse
import pandas as pd
from skimage import io
import cupy as cp
from cucim.skimage.measure import regionprops
from tqdm import tqdm
import numpy as np


"""
Description: This script extracts features from label images and saves them to a csv file. 
It is possible to quantify intensity by specifying the channel. 
It uses the cucim library to speed up the process.

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Get some regionprops of all objects in all tifs in a input_folder.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--label_pattern", type=str, default='_labels.tif', help="Pattern to identify label images.")
    # if intensity quantification, ask for the channel. Default is no intensity quantification
    parser.add_argument("--channel", type=int, default=-1, help="Channel to quantify intensity.")
    return parser.parse_args()

args = parse_args()

input_folder = args.input
label_pattern = args.label_pattern
channel = args.channel

def get_regionprops(label_img_path, intensity_img_path=None):
    
    df = pd.DataFrame()
    label_img = cp.asarray(io.imread(label_img_path))
    
    if channel != -1:
        intensity_img_np = io.imread(intensity_img_path)

        # Handle 3D images
        if intensity_img_np.ndim == 2 or intensity_img_np.ndim == 3:
            # For 3D images, no channel selection is needed
            intensity_img = cp.asarray(intensity_img_np)
        elif intensity_img_np.ndim == 4:
            # For 4D images (e.g., ZYX with a channel), select the specified channel
            intensity_img = cp.asarray(intensity_img_np[:,:,:,channel])
        else:
            raise ValueError(f"Unexpected number of dimensions: {intensity_img_np.ndim}")
        
        props = regionprops(label_img, intensity_img)
        for i, prop in enumerate(props):
            df.loc[i, 'Filename'] = os.path.basename(label_img_path)
            df.loc[i, 'Labels_ID'] = int(prop.label) 
            df.loc[i, 'Area'] = prop.area.get()
            #df.loc[i, 'Perimeter'] = prop.perimeter.get()
            
            # Check if the image is 3D and skip eccentricity
            if label_img.ndim == 3:
                df.loc[i, 'Eccentricity'] = np.nan  # Set to NaN for 3D images
            else:
                df.loc[i, 'Eccentricity'] = prop.eccentricity
            
            df.loc[i, 'MajorAxisLength'] = prop.major_axis_length
            df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length
            df.loc[i, 'MeanIntensity'] =  prop.intensity_mean.get()
            df.loc[i, 'MaxIntensity'] =  prop.intensity_max.get()
    
    else:
        props = regionprops(label_img)
        for i, prop in enumerate(props):
            df.loc[i, 'Filename'] = os.path.basename(label_img_path)
            df.loc[i, 'Labels_ID'] = int(prop.label) 
            df.loc[i, 'Area'] = prop.area.get()
            #df.loc[i, 'Perimeter'] = prop.perimeter.get()
            
            # Check if the image is 3D and skip eccentricity
            if label_img.ndim == 3:
                df.loc[i, 'Eccentricity'] = np.nan  # Set to NaN for 3D images
            else:
                df.loc[i, 'Eccentricity'] = prop.eccentricity
            
            df.loc[i, 'MajorAxisLength'] = prop.major_axis_length
            df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length

    return df



def main():
    # Get the label images
    label_imgs = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif') and f.endswith(label_pattern)]
    label_imgs = sorted(label_imgs)
   
   
   
    # Create an empty dataframe to store the regionprops
    regionprops_df = pd.DataFrame() 
    
    if channel != -1:    
        # get intensities
        intensity_imgs = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith(label_pattern)]
        intensity_imgs = sorted(intensity_imgs)
    
        # check if the number of label images is the same as the number of intensity images
        if len(label_imgs) != len(intensity_imgs):
            raise ValueError('The number of label images is not the same as the number of intensity images.')

        for label_img_path, intensity_img_path in tqdm(zip(label_imgs, intensity_imgs), total=len(label_imgs), desc="Processing images"):

            # Get the regionprops
            regionprops = get_regionprops(label_img_path, intensity_img_path)

            # Add the regionprops to the dataframe by concatenating
            regionprops_df = pd.concat([regionprops_df, regionprops], ignore_index=True)
    else:
        for label_img_path in tqdm(label_imgs, total=len(label_imgs), desc="Processing images"):

            # Get the regionprops
            regionprops = get_regionprops(label_img_path)

            # Add the regionprops to the dataframe by concatenating
            regionprops_df = pd.concat([regionprops_df, regionprops], ignore_index=True)    
    
        
    # Save the regionprops to a csv file
    regionprops_df.to_csv(os.path.join(input_folder, 'regionprops.csv'), index=False)
    

if __name__ == '__main__':
    main()
