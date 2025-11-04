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
Description: This script measures properties within ROIs defined by Channel 1 labels.

Two modes:
1. Label-based: Measures properties of Channel 2 labeled objects that overlap with Channel 1 ROIs
   - Use when you have segmented objects in both channels
   - Example: Count nuclei within cell ROIs and measure their properties
   
2. Intensity-based: Measures intensity statistics directly from Channel 2 intensity image within Channel 1 ROIs
   - Use when you only want to measure intensity, no object segmentation needed in Channel 2
   - Example: Measure mean fluorescence intensity within segmented cells
   - Outputs: Mean, Median, Std, Max, Min intensity per ROI

Example Usage:

# Mode 1: Label-based (original functionality)
python ROI_colocalization_regionprops.py --input /path/to/data \
    --channels FITC DAPI \
    --channel1_pattern "*_labels.tif" \
    --channel2_pattern "*_labels.tif" \
    --channel2_is_labels y \
    --label_ids 1 2 3 \
    --ROI_size n

# Mode 2: Intensity-based (new functionality - simpler!)
python ROI_colocalization_regionprops.py --input /path/to/data \
    --channels CellMask GFP \
    --channel1_pattern "*_labels.tif" \
    --channel2_pattern "*.tif" \
    --channel2_is_labels n \
    --label_ids 1 2 3 \
    --ROI_size y

# Analyze ALL ROIs automatically (omit --label_ids)
python ROI_colocalization_regionprops.py --input /path/to/data \
    --channels CellMask GFP \
    --channel1_pattern "*_labels.tif" \
    --channel2_pattern "*.tif" \
    --channel2_is_labels n \
    --ROI_size y

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for obtaining regionprops within ROI. Supports both label-based and intensity-based measurements.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of the two color channels. Example: "TRITC DAPI"')
    parser.add_argument('--channel1_pattern', type=str, required=True, help='Label pattern for channel 1 (ROI labels). Example: "*_labels.tif"')
    parser.add_argument('--channel2_pattern', type=str, required=True, help='File pattern for channel 2. Can be labels (*_labels.tif) or intensity images (*.tif)')
    parser.add_argument('--channel2_is_labels', type=str, default='y', help='Is channel 2 a label image? (y/n). If "n", will measure intensity directly without object segmentation.')
    # select label ids in first channel
    parser.add_argument('--label_ids', nargs='+', type=int, required=False, default=None, help='Label ids to be analyzed in the first channel. If not specified, all ROIs will be analyzed automatically.')
    parser.add_argument('--ROI_size', type=str, default='n', help='Do you want to get size of ROIs in the first channel? (y/n)')
    return parser.parse_args()

def median_intensity(label_img, intensity_img):
        return cp.median(intensity_img[label_img])

def std_intensity(label_img, intensity_img):
        return cp.std(intensity_img[label_img])

def get_regionprops(label_img, intensity_img, file_path, label_id,channels, ROI_size='n',c1_regionprops=None):
    df = pd.DataFrame()
    
    # Check if shapes match
    if label_img.shape != intensity_img.shape:
        raise ValueError(
            f"Shape mismatch in {os.path.basename(file_path)}:\n"
            f"  Channel 2 label image shape: {label_img.shape}\n"
            f"  Channel 2 intensity image shape: {intensity_img.shape}\n"
            f"Both images must have the same dimensions. Please ensure your images are properly aligned."
        )
    
    label_img = cp.asarray(label_img)
    intensity_img = cp.asarray(intensity_img)
    props = regionprops(label_img, intensity_img, extra_properties=[median_intensity, std_intensity])
    for i, prop in enumerate(props):
        df.loc[i, 'Filename'] = os.path.basename(file_path)
        df.loc[i, f'{channels[0]} label id'] = label_id
        if ROI_size.lower() == 'y':
            # Find the regionprop that matches the label_id (not using label_id as index)
            roi_prop = next((p for p in c1_regionprops if p.label == label_id), None)
            if roi_prop is not None:
                df.loc[i, f'{channels[0]} Size'] = roi_prop.area  # area = pixel/voxel count (2D/3D)
        else:
            pass
        df.loc[i, f'{channels[1]} label id'] = int(prop.label)
        try:
            df.loc[i, f'{channels[1]} Size'] = prop.area.get()  # area = pixel/voxel count (2D/3D)
        except ValueError:
            print(f"Skipping size for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} Size'] = np.nan
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
            df.loc[i, f'{channels[1]} MajorAxisLength'] = prop.major_axis_length
        except ValueError:
            print(f"Skipping major axis length for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} MajorAxisLength'] = np.nan
        try:
            df.loc[i, f'{channels[1]} MinorAxisLength'] = prop.minor_axis_length
        except ValueError:
            print(f"Skipping minor axis length for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} MinorAxisLength'] = np.nan
        try:
            df.loc[i, f'{channels[1]} MeanIntensity'] = prop.intensity_mean.get()
        except ValueError:
            print(f"Skipping mean intensity for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} MeanIntensity'] = np.nan
        try:
            df.loc[i, f'{channels[1]} MedianIntensity'] = prop.median_intensity.get()
        except ValueError:
            print(f"Skipping median intensity for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} MedianIntensity'] = np.nan
        try:
            df.loc[i, f'{channels[1]} MaxIntensity'] = prop.intensity_max.get()
        except ValueError:
            print(f"Skipping max intensity for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} MaxIntensity'] = np.nan
        try:
            df.loc[i, f'{channels[1]} StdIntensity'] = prop.std_intensity.get()
        except ValueError:
            print(f"Skipping std intensity for region {prop.label} due to numerical error.")
            df.loc[i, f'{channels[1]} StdIntensity'] = np.nan

    return df


def get_intensity_only_regionprops(ROI_mask, intensity_img, file_path, label_id, channels, ROI_size):
    """
    Measure intensity statistics directly from intensity image within ROI mask.
    No object segmentation in channel 2 - just pure intensity measurements.
    """
    df = pd.DataFrame()
    
    # Check if shapes match
    if ROI_mask.shape != intensity_img.shape:
        raise ValueError(
            f"Shape mismatch in {os.path.basename(file_path)}:\n"
            f"  Channel 1 ROI mask shape: {ROI_mask.shape}\n"
            f"  Channel 2 intensity image shape: {intensity_img.shape}\n"
            f"Both images must have the same dimensions. Please ensure your images are properly aligned."
        )
    
    # Create a binary mask for the ROI
    ROI_mask_gpu = cp.asarray(ROI_mask)
    intensity_img_gpu = cp.asarray(intensity_img)
    
    # Get intensity values within the ROI
    intensity_values = intensity_img_gpu[ROI_mask_gpu > 0]
    
    # Only create entry if ROI has pixels
    if len(intensity_values) > 0:
        df.loc[0, 'Filename'] = os.path.basename(file_path)
        df.loc[0, f'{channels[0]} label id'] = label_id
        if ROI_size.lower() == 'y':
            # Calculate ROI size in pixels/voxels (works for both 2D and 3D)
            roi_size = int(cp.sum(ROI_mask_gpu > 0).get())
            df.loc[0, f'{channels[0]} Size'] = roi_size
        
        # Intensity measurements
        df.loc[0, f'{channels[1]} MeanIntensity'] = float(cp.mean(intensity_values).get())
        df.loc[0, f'{channels[1]} MedianIntensity'] = float(cp.median(intensity_values).get())
        df.loc[0, f'{channels[1]} StdIntensity'] = float(cp.std(intensity_values).get())
        df.loc[0, f'{channels[1]} MaxIntensity'] = float(cp.max(intensity_values).get())
        df.loc[0, f'{channels[1]} MinIntensity'] = float(cp.min(intensity_values).get())
    
    return df


def coloc_channels(file_lists, channels, parent_dir, label_ids, ROI_size, channel2_is_labels):
    
    
    # Create an empty dataframe to store the regionprops
    regionprops_df = pd.DataFrame() 

    # Create a list of file paths for the first channel
    file_paths = file_lists[channels[0]]

    # Wrap the loop with tqdm
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Processing images"):

        image_c1 = imread(file_lists[channels[0]][file_paths.index(file_path)])
        image_c2 = imread(file_lists[channels[1]][file_paths.index(file_path)])
        
        # Only compute c1_regionprops if needed (for label-based mode with ROI_size)
        c1_regionprops = None
        if ROI_size.lower() == 'y' and channel2_is_labels.lower() == 'y':
            c1_regionprops = regionprops(image_c1)

        # Determine if we need to load separate intensity image
        if channel2_is_labels.lower() == 'y':
            # Original behavior: load separate intensity image for labeled objects
            image_c2_intensities = imread(file_lists[channels[1]][file_paths.index(file_path)].replace('_labels.tif', '.tif'))
        else:
            # New behavior: use image_c2 directly as intensity image
            image_c2_intensities = image_c2

        # Determine which ROIs to analyze
        if label_ids is None:
            # Auto-detect all ROIs in the image
            current_label_ids = np.unique(image_c1)
            current_label_ids = current_label_ids[current_label_ids != 0]  # drop the background label (0)
            if len(current_label_ids) > 0:
                print(f"Auto-detected {len(current_label_ids)} ROIs in {os.path.basename(file_path)}: {list(current_label_ids)}")
        else:
            # Use user-specified ROIs
            current_label_ids = label_ids

        for label_id in current_label_ids:
            ROI_mask = image_c1 == label_id # boolean mask with true where label_id is present in image_c1
            
            if channel2_is_labels.lower() == 'y':
                # Original behavior: measure properties of labeled objects in channel 2 within ROI
                c2_in_c1 = image_c2 * ROI_mask
                # c2_in_c1 should now only contain the label_id objects that are inside the ROI_mask
                all_regionprops = get_regionprops(c2_in_c1, image_c2_intensities, file_path, label_id, channels, ROI_size, c1_regionprops)
            else:
                # New behavior: measure intensity directly within ROI (no object segmentation needed)
                all_regionprops = get_intensity_only_regionprops(ROI_mask, image_c2_intensities, file_path, label_id, channels, ROI_size)
            
            # Add the regionprops to the dataframe by concatenating
            regionprops_df = pd.concat([regionprops_df, all_regionprops], ignore_index=True)

            
    # Save the regionprops to a csv file with channel names in filename
    output_filename = f'{channels[0]}_{channels[1]}_ROI_regionprops.csv'
    regionprops_df.to_csv(os.path.join(parent_dir, output_filename), index=False)


def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    channel1_pattern = args.channel1_pattern
    channel2_pattern = args.channel2_pattern
    channel2_is_labels = args.channel2_is_labels
    label_ids = args.label_ids 
    ROI_size = args.ROI_size

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    # Build file lists for both channels
    file_lists = {
        channels[0]: sorted(glob.glob(os.path.join(parent_dir, channels[0], channel1_pattern))),
        channels[1]: sorted(glob.glob(os.path.join(parent_dir, channels[1], channel2_pattern)))
    }
    
    # Verify we have matching files
    if len(file_lists[channels[0]]) == 0:
        raise ValueError(f"No files found for channel 1 with pattern: {channel1_pattern}")
    if len(file_lists[channels[1]]) == 0:
        raise ValueError(f"No files found for channel 2 with pattern: {channel2_pattern}")
    if len(file_lists[channels[0]]) != len(file_lists[channels[1]]):
        raise ValueError(f"Mismatch in file counts: Channel 1 has {len(file_lists[channels[0]])} files, Channel 2 has {len(file_lists[channels[1]])} files")

    # Print analysis configuration
    print("\n" + "="*60)
    print("ROI Regionprops Analysis Configuration")
    print("="*60)
    print(f"Input folder: {parent_dir}")
    print(f"Channel 1 (ROI): {channels[0]} - {len(file_lists[channels[0]])} files")
    print(f"Channel 2 (Measurement): {channels[1]} - {len(file_lists[channels[1]])} files")
    print(f"Mode: {'Label-based' if channel2_is_labels.lower() == 'y' else 'Intensity-based'}")
    if label_ids is None:
        print(f"ROI Selection: Auto-detect all ROIs in each image")
    else:
        print(f"ROI Selection: Specified ROIs {label_ids}")
    print(f"Include ROI sizes: {ROI_size.upper()}")
    print("="*60 + "\n")

    coloc_channels(file_lists, channels, parent_dir, label_ids, ROI_size, channel2_is_labels)



if __name__ == "__main__":
    main()
