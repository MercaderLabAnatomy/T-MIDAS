import os
import glob
import csv
from skimage import io
import argparse
import numpy as np
import cupy as cp
from cucim.skimage.measure import label, regionprops
from tifffile import imwrite

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    parser.add_argument('--output_images', type=str, help='Do you want to save colocalization images? (y/n)')
    return parser.parse_args()

args = parse_arguments()


parent_dir = args.input + '/'
channels = [c.upper() for c in args.channels]
label_patterns = args.label_patterns

# parent_dir = "/media/geffjoldblum/DATA/ImagesJoao"
# channels = ["FITC", "CY5"]
# label_patterns = ["*_labels.tif", "*_labels.tif"]



# Get a list of files for each channel
file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")

print("Number of label files in each channel:")
{print(channel, ":", len(file_lists[channel])) for channel in file_lists}

# Perform colocalization analysis for specified channels
def coloc_channels(file_lists, channels):
    csv_rows = []
    
    for i in range(len(file_lists[channels[0]])):
        images = {channel: cp.asarray(io.imread(file_lists[channel][i])) for channel in channels}
        
        ROI_masks = {label_id.item(): images[channels[0]] == label_id for label_id in cp.unique(images[channels[0]]) if label_id != 0}
        
        overlaps = {channel: {} for channel in channels}
        unique_labels = {channel: {} for channel in channels} #

        area = {label_id: regionprops(ROI_masks[label_id].astype(np.int32))[0].area for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        
        for idx, channel in enumerate(channels):
            other_channels = [c for c in channels if c != channel]
            for label_id in ROI_masks.keys():
                if ROI_masks[label_id] is not None: # binary mask of reference channel
                    # try catch if shapes do not match
                    # get shapes of ROI_masks[label_id] and images[other_channels[0]]
                    # if they do not match, print filename  
                    # if they do match, continue with the rest of the code
                    # get the regionprops area of the ROI_masks[label_id]
                    
                    if ROI_masks[label_id].shape != images[other_channels[0]].shape:
                        print(f"Shapes do not match for {file_lists[channels[0]][i]}")
                        continue
                    else:
                        pass
                    
                    overlaps[channel][label_id] = ROI_masks[label_id] & (images[other_channels[0]] > 0)
                    unique_labels[channel][label_id] = cp.max(cp.unique(label(overlaps[channel][label_id]))).item()
                    
                    if len(channels) > 2:
                        for other_channel in other_channels[1:]:
                            overlaps[channel][label_id] &= (images[other_channel] > 0)
                            unique_labels[channel][label_id] += cp.max(cp.unique(label(overlaps[channel][label_id]))).item()
        
        filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]
        
        if args.output_images.lower() == 'y':
            if len(channels) == 2:
                for label_id in unique_labels[channels[0]].keys():
                    colocalization_image = label(cp.asarray(overlaps[channels[0]][label_id]))
                    #colocalization_image = cp.where(colocalization_image, 255, 0).astype(cp.uint8) # this is to
                    imwrite(parent_dir + f"/{filename}_{channels[1]}_in_{channels[0]}_ROI_{label_id}.tif", cp.asnumpy(colocalization_image), compression='zlib')
            elif len(channels) == 3:
                for label_id in unique_labels[channels[0]].keys():
                    colocalization_image = label(cp.asarray(overlaps[channels[0]][label_id]))
                    #colocalization_image = cp.where(colocalization_image, 255, 0).astype(cp.uint8)
                    imwrite(parent_dir + f"/{filename}_{channels[1]}_{channels[2]}_coloc_in_{channels[0]}_ROI_{label_id}.tif", cp.asnumpy(colocalization_image), compression='zlib')
            else:
                raise ValueError("Only two or three channels are supported for saving colocalization images.")

                

        
        

        # if two channels are provided, save a row for each first channel ROI that contains filename, first channel ROI id and colocalization count
        if len(channels) == 2:
            for label_id in unique_labels[channels[0]].keys():
                csv_rows.append([filename, label_id, area[label_id], unique_labels[channels[0]][label_id]])
        elif len(channels) == 3:
            for label_id in unique_labels[channels[0]].keys():
                csv_rows.append([filename, label_id, area[label_id], *[unique_labels[channel][label_id] for channel in channels[1:]], unique_labels[channels[0]][label_id]])

        
        cp.get_default_memory_pool().free_all_blocks()
    
    return csv_rows

# Check the number of channels and perform colocalization analysis accordingly
csv_rows = coloc_channels(file_lists, channels)

csv_file = parent_dir + '/colocalization.csv'

# Write results to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    if len(channels) == 2:
        writer.writerow(['Filename', f"{channels[0]} ROI", "Area (sq. px)", 
                         *[f"{channel}_in_{channels[0]}" for channel in channels[1:]]])
    elif len(channels) == 3:
        writer.writerow(['Filename', f"{channels[0]} ROI", "Area (sq. px)", 
                         *[f"{channel}_in_{channels[0]}" for channel in channels[1:]], 
                         f"{channels[2]}_in_{channels[1]}_in_{channels[0]}"])
    writer.writerows(csv_rows)

print(f"Colocalization results saved to {csv_file}")
print("Done!")
