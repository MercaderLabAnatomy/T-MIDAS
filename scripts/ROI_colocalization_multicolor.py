import os
import glob
import csv
from skimage import io
import argparse
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix, coo_matrix
from cucim.skimage.measure import label, regionprops
from tifffile import imwrite

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
        
        label_ids = cp.unique(images[channels[0]])
        ROI_masks = {}

        for label_id in label_ids:
            ROI_masks[label_id.item()] = csr_matrix(images[channels[0]] == label_id)
        coloc_01 = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        coloc_02 = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        coloc_all = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        unique_labels_01 = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        unique_labels_02 = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        unique_labels_all = {label_id: {} for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}  
        area = {label_id: regionprops(ROI_masks[label_id].toarray().astype(np.int32))[0].area for label_id in ROI_masks.keys() if ROI_masks[label_id] is not None}
        
        for label_id in ROI_masks.keys():
            if ROI_masks[label_id] is not None:              
                coloc_01[label_id] = (ROI_masks[label_id] != csr_matrix(images[channels[1]] > 0)).toarray()          
                unique_labels_01[label_id] = cp.max(cp.unique(label(coloc_01[label_id]))).item()
                
                if len(channels) == 3:
                    coloc_02[label_id] = (ROI_masks[label_id] != csr_matrix(images[channels[2]] > 0)).toarray()
                    coloc_all[label_id] = (ROI_masks[label_id] != (csr_matrix((images[channels[1]] > 0) & images[channels[2]] > 0))).toarray()                   
                    unique_labels_02[label_id] = cp.max(cp.unique(label(coloc_02[label_id]))).item()
                    unique_labels_all[label_id] = cp.max(cp.unique(label(coloc_all[label_id]))).item()
        
        filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]
        
        if args.output_images.lower() == 'y' and len(label_ids) <= 8:
            if len(channels) == 2:
                for label_id in unique_labels_01.keys():
                    coloc_01_image = label(cp.asarray(coloc_01[label_id]))
                    imwrite(parent_dir + f"/{filename}_{channels[1]}_in_{channels[0]}_ROI_{label_id}.tif", 
                            cp.asnumpy(coloc_01_image), compression='zlib')
                    
            elif len(channels) == 3 and len(label_ids) <= 8:
                for label_id in unique_labels_02.keys():
                    coloc_02_image = label(cp.asarray(coloc_02[label_id]))
                    imwrite(parent_dir + f"/{filename}_{channels[2]}_in_{channels[0]}_ROI_{label_id}.tif", 
                            cp.asnumpy(coloc_02_image), compression='zlib')
                for label_id in unique_labels_all.keys():
                    coloc_all_image = label(cp.asarray(coloc_all[label_id]))
                    imwrite(parent_dir + f"/{filename}_{channels[1]}_{channels[2]}_coloc_in_{channels[0]}_ROI_{label_id}.tif", 
                            cp.asnumpy(coloc_all_image), compression='zlib')
            else:
                raise ValueError("Only two or three channels are supported for saving colocalization images.")

        if len(channels) == 2:
            for label_id in unique_labels_01.keys():
                csv_rows.append([filename, label_id, area[label_id], unique_labels_01[label_id]])
        elif len(channels) == 3:
            for label_id in unique_labels_all.keys():
                csv_rows.append([filename, label_id, area[label_id], unique_labels_01[label_id], unique_labels_02[label_id], unique_labels_all[label_id]])

        cp.get_default_memory_pool().free_all_blocks()
    
    return csv_rows

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
