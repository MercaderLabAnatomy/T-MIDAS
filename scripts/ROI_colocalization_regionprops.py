import os
import glob
import argparse
import numpy as np
import pandas as pd
import cupy as cp
from cucim.skimage.measure import regionprops
from skimage.io import imread
from tqdm import tqdm


"""
Description: This script reads label images from two color channels and 
calculates regionprops of objects in the second channel that are inside ROIs in the first channel.

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for obtaining regionprops of objects in second channel in ROIs of first channel.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of the two color channels. Example: "TRITC DAPI"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif "')
    # select label ids in first channel
    parser.add_argument('--label_ids', nargs='+', type=int, required=True, help='Label ids to be analyzed in the first channel.')
    parser.add_argument('--ROI_size', type=str, default='n', help='Do you want to get size of ROIs in the first channel? (y/n)')
    return parser.parse_args()

def median_intensity(label_img, intensity_img):
        return cp.median(intensity_img[label_img])

def get_regionprops(label_img, intensity_img, file_path, label_id,channels, ROI_size='n',c1_regionprops=None):
    df = pd.DataFrame()
    label_img = cp.asarray(label_img)
    intensity_img = cp.asarray(intensity_img)
    props = regionprops(label_img, intensity_img, extra_properties=[median_intensity])
    for i, prop in enumerate(props):
        df.loc[i, 'Filename'] = os.path.basename(file_path)
        df.loc[i, f'{channels[0]} label id'] = label_id
        if ROI_size.lower() == 'y':
            df.loc[i, f'{channels[0]} Size'] = c1_regionprops[label_id-1].area
        else:
            pass
        df.loc[i, f'{channels[1]} label id'] = int(prop.label)
        try:
            df.loc[i, 'Size'] = prop.area.get()
        except ValueError:
            print(f"Skipping size for region {prop.label} due to numerical error.")
            df.loc[i, 'Size'] = np.nan
        # try:
        #     df.loc[i, 'Perimeter'] = prop.perimeter.get()
        # except ValueError:
        #     print(f"Skipping perimeter for region {prop.label} due to numerical error.")
        #     df.loc[i, 'Perimeter'] = np.nan
        # try:
        #     df.loc[i, 'Eccentricity'] = prop.eccentricity
        # except ValueError:
        #     print(f"Skipping eccentricity for region {prop.label} due to numerical error.")
        #     df.loc[i, 'Eccentricity'] = np.nan
        try:
            df.loc[i, 'MajorAxisLength'] = prop.major_axis_length
        except ValueError:
            print(f"Skipping major axis length for region {prop.label} due to numerical error.")
            df.loc[i, 'MajorAxisLength'] = np.nan
        try:
            df.loc[i, 'MinorAxisLength'] = prop.minor_axis_length
        except ValueError:
            print(f"Skipping minor axis length for region {prop.label} due to numerical error.")
            df.loc[i, 'MinorAxisLength'] = np.nan
        try:
            df.loc[i, 'MeanIntensity'] = prop.intensity_mean.get()
        except ValueError:
            print(f"Skipping mean intensity for region {prop.label} due to numerical error.")
            df.loc[i, 'MeanIntensity'] = np.nan
        try:
            df.loc[i, 'MedianIntensity'] = prop.median_intensity.get()
        except ValueError:
            print(f"Skipping median intensity for region {prop.label} due to numerical error.")
            df.loc[i, 'MedianIntensity'] = np.nan
        try:
            df.loc[i, 'MaxIntensity'] = prop.intensity_max.get()
        except ValueError:
            print(f"Skipping max intensity for region {prop.label} due to numerical error.")
            df.loc[i, 'MaxIntensity'] = np.nan

    return df





def coloc_channels(file_lists, channels, parent_dir,label_ids, ROI_size):
    
    
    # Create an empty dataframe to store the regionprops
    regionprops_df = pd.DataFrame() 

    # Create a list of file paths for the first channel
    file_paths = file_lists[channels[0]]

    # Wrap the loop with tqdm
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Processing images"):

        image_c1 = imread(file_lists[channels[0]][file_paths.index(file_path)])
        image_c2 = imread(file_lists[channels[1]][file_paths.index(file_path)])
        
        c1_regionprops = None

        if ROI_size.lower() == 'y':
            c1_regionprops = regionprops(image_c1)
        else:
            pass

        # get intensities of image_c2 by removing the label_pattern
        image_c2_intensities = imread(file_lists[channels[1]][file_paths.index(file_path)].replace('_labels.tif', '.tif'))

        # label_ids = np.unique(image_c1)
        # label_ids = label_ids[label_ids != 0] # drop the background label

        for label_id in label_ids:
            ROI_mask = image_c1 == label_id # boolean mask with true where label_id is present in image_c1
            # now select the subset of c2 objects in ROI_mask
            c2_in_c1 = image_c2 * ROI_mask
            # c2_in_c1 should now only contain the label_id objects that are inside the ROI_mask

            # Get the regionprops
            all_regionprops = get_regionprops(c2_in_c1, image_c2_intensities,file_path, label_id,channels, ROI_size,c1_regionprops)
            # Add the regionprops to the dataframe by concatenating
            regionprops_df = pd.concat([regionprops_df, all_regionprops], ignore_index=True)

            
    # Save the regionprops to a csv file
    regionprops_df.to_csv(os.path.join(parent_dir, 'ROI_colocalization_regionprops.csv'), index=False)


def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    label_patterns = args.label_patterns
    label_ids = args.label_ids 
    ROI_size = args.ROI_size

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

    coloc_channels(file_lists, channels, parent_dir,label_ids, ROI_size)



if __name__ == "__main__":
    main()
