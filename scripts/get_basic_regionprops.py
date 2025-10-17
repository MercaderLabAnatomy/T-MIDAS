import os
import argparse
import sys
import pandas as pd
from skimage import io
import cupy as cp
from cucim.skimage.measure import regionprops
from tqdm import tqdm
import numpy as np

# Add tmidas to path
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.argparse_utils import create_parser


"""
Description: This script extracts features from label images and saves them to a csv file. 
It is possible to quantify intensity by specifying the channel. 
It uses the cucim library to speed up the process.

Pixel Resolution:
- Use --pixel_size_xy to specify the pixel size in XY plane (e.g., 0.325 for 0.325 µm/pixel)
- Use --pixel_size_z to specify the pixel size in Z dimension for 3D images (e.g., 1.0 for 1.0 µm/slice)
- Use --pixel_unit to specify the unit (default: 'µm')
- When pixel resolution is provided, Area, MajorAxisLength, and MinorAxisLength will be converted to physical units
- Output CSV will include unit columns (Area_Unit, MajorAxisLength_Unit, MinorAxisLength_Unit)

"""

def parse_args():
    parser = create_parser("Get some regionprops of all objects in all tifs in a input_folder.")
    parser.add_argument("--label_pattern", type=str, default='_labels.tif', help="Pattern to identify label images.")
    # if intensity quantification, ask for the channel. Default is no intensity quantification
    parser.add_argument("--channel", type=int, default=-1, help="Channel to quantify intensity.")
    # Pixel resolution arguments
    parser.add_argument("--pixel_size_xy", type=float, default=None, help="Pixel size in XY plane (e.g., 0.325 for 0.325 µm/pixel).")
    parser.add_argument("--pixel_size_z", type=float, default=None, help="Pixel size in Z dimension (e.g., 1.0 for 1.0 µm/slice). Only needed for 3D images.")
    parser.add_argument("--pixel_unit", type=str, default="µm", help="Unit of pixel size (e.g., 'µm', 'nm', 'mm'). Default is 'µm'.")
    return parser.parse_args()

args = parse_args()

input_folder = args.input
label_pattern = args.label_pattern
channel = args.channel
pixel_size_xy = args.pixel_size_xy
pixel_size_z = args.pixel_size_z
pixel_unit = args.pixel_unit

def get_regionprops(label_img_path, intensity_img_path=None):
    
    df = pd.DataFrame()
    label_img = cp.asarray(io.imread(label_img_path))
    
    # Determine if we have pixel resolution info
    has_resolution = pixel_size_xy is not None
    is_3d = label_img.ndim == 3
    
    # Calculate area scaling factor
    if has_resolution:
        if is_3d and pixel_size_z is not None:
            # For 3D: volume = pixels * (xy_size^2) * z_size
            area_scale = (pixel_size_xy ** 2) * pixel_size_z
            length_scale = pixel_size_xy
            area_unit = f"{pixel_unit}³"
            length_unit = pixel_unit
        else:
            # For 2D: area = pixels * (xy_size^2)
            area_scale = pixel_size_xy ** 2
            length_scale = pixel_size_xy
            area_unit = f"{pixel_unit}²"
            length_unit = pixel_unit
    else:
        area_scale = 1.0
        length_scale = 1.0
        area_unit = "pixels³" if is_3d else "pixels²"
        length_unit = "pixels"
    
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
            df.loc[i, 'Area'] = prop.area.get() * area_scale
            df.loc[i, 'Area_Unit'] = area_unit
            #df.loc[i, 'Perimeter'] = prop.perimeter.get()
            
            # Check if the image is 3D and skip eccentricity
            if label_img.ndim == 3:
                df.loc[i, 'Eccentricity'] = np.nan  # Set to NaN for 3D images
            else:
                df.loc[i, 'Eccentricity'] = prop.eccentricity
            
            df.loc[i, 'MajorAxisLength'] = prop.major_axis_length * length_scale
            df.loc[i, 'MajorAxisLength_Unit'] = length_unit
            df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length * length_scale
            df.loc[i, 'MinorAxisLength_Unit'] = length_unit
            df.loc[i, 'MeanIntensity'] =  prop.intensity_mean.get()
            df.loc[i, 'MaxIntensity'] =  prop.intensity_max.get()
    
    else:
        props = regionprops(label_img)
        for i, prop in enumerate(props):
            df.loc[i, 'Filename'] = os.path.basename(label_img_path)
            df.loc[i, 'Labels_ID'] = int(prop.label) 
            df.loc[i, 'Area'] = prop.area.get() * area_scale
            df.loc[i, 'Area_Unit'] = area_unit
            #df.loc[i, 'Perimeter'] = prop.perimeter.get()
            
            # Check if the image is 3D and skip eccentricity
            if label_img.ndim == 3:
                df.loc[i, 'Eccentricity'] = np.nan  # Set to NaN for 3D images
            else:
                df.loc[i, 'Eccentricity'] = prop.eccentricity
            
            df.loc[i, 'MajorAxisLength'] = prop.major_axis_length * length_scale
            df.loc[i, 'MajorAxisLength_Unit'] = length_unit
            df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length * length_scale
            df.loc[i, 'MinorAxisLength_Unit'] = length_unit

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
