import os
import glob
import csv
from skimage import io
from skimage.measure import regionprops
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--add_intensity', type=str, help='Do you want to quantify average intensity of C2 in C1 ROI? (y/n)')
    parser.add_argument('--label_patterns', nargs='+', type=str, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    return parser.parse_args()

args = parse_arguments()

parent_dir = args.input + '/'
channels = [c.upper() for c in args.channels]
label_patterns = [p for p in args.label_patterns]

if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")

def get_file_list(parent_dir, channels, label_patterns):
    file_lists = {}
    
    for channel, label_pattern in zip(channels, label_patterns):
        labels = sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern)))
        file_lists[channel] = labels
        
    return file_lists

file_lists = get_file_list(parent_dir, channels, label_patterns)

print("Number of label files in each channel:")
for channel in file_lists.keys():
    print(channel, ":", len(file_lists[channel]))

lengths = [len(file_lists[channel]) for channel in file_lists.keys()]
if len(set(lengths)) > 1:
    print("Warning: The number of label files in the different channels is not the same.")
    print("Number of label files in each channel:", lengths)

csv_rows = []

def get_coords_mask(prop, shape):
    # using prop.coords is more efficient than prop.image
    x_coords, y_coords = prop.coords[:, 0], prop.coords[:, 1]
    mask = np.zeros(shape, dtype=bool)
    mask[x_coords, y_coords] = True

    return mask


def coloc_channels(file_lists, channels, csv_rows, add_intensity=False):
    num_channels = len(channels)
    
    for i in range(len(file_lists[channels[0]])):
        images = [io.imread(file_lists[channel][i]) for channel in channels]
        props = [regionprops(img) for img in images]
        
        filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]

        for prop0 in props[0]: # this is the first channel
            area = prop0.area
            if 100 < area:# < 100000:
                mask0 = get_coords_mask(prop0, images[0].shape)

                centroid_in_regions = [False] * num_channels
                mean_intensities = [None] * num_channels

                for idx, (prop, img) in enumerate(zip(props[1:], images[1:])):
                    mask = get_coords_mask(prop[0], img.shape)
                    centroid = prop.centroid
                    row, col = int(centroid[0]), int(centroid[1])

                    mean_intensity_in_region = np.mean(img[mask])
                    if mask0[row, col]:
                        centroid_in_regions[idx + 1] = True
                        mean_intensities[idx + 1] = mean_intensity_in_region

                row_data = [filename, prop0.label, area] + centroid_in_regions + mean_intensities
                csv_rows.append(row_data)

args.add_intensity = args.add_intensity.lower()
output_csv = os.path.join(parent_dir, 'colocalization.csv')

header_base = ['Filename', f'{channels[0]}_label', f'{channels[0]}_area']
header_cols = [f'{channel}_centroid_in_{channels[0]}_region' for channel in channels[1:]]
header_intensities = [f'{channel}_mean_intensity_in_{channels[0]}_region' for channel in channels[1:]]

header = header_base + header_cols + header_intensities if args.add_intensity == 'y' else header_base + header_cols



if os.path.exists(output_csv):
    os.remove(output_csv)

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_rows)

print("Colocalization data saved to", output_csv)
print("Done.")
