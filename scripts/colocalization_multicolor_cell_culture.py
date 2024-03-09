import os
import glob
import csv
from skimage import io
from skimage.measure import regionprops
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Count proliferating FITC+ cells.')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels',  nargs='+', type=str, help='Names of all color channels. Example: "TRITC DAPI FITC"')
    # name of color channel that is supposed to serve as target
    parser.add_argument('--target', type=str, help='Name of the target channel. Bounding boxes of all objects in this channel will be checked against centroids of all objects in the other color channels. Example: "FITC"')
    parser.add_argument('--label_pattern', type=str, default='*_labels.tif', help='Pattern to match label images. Default: "*_labels.tif"')
    return parser.parse_args()


args = parse_arguments()
# output of parse_arguments is a Namespace object
# a namespace object can be accessed like a dictionary

# Define the naming pattern for label and intensity images
label_pattern = args.label_pattern
parent_dir = args.input +'/'
channels = [c.upper() for c in args.channels]

# give error if channel names are not unique or smaller than 2
if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")



target = args.target

#channels = ["TRITCA", "CY5"]
# target = "FITC"
# parent_dir = "/media/geffjoldblum/DATA/images_joao/20240205_ntn1a_col1a2_IB4"
# label_pattern = "*_labels.tif"

def get_file_list(parent_dir, channels, label_pattern):
    file_lists = {}  # Dictionary to store lists with channel names as keys

    for channel in channels:
        labels = sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern)))
        file_lists[channel] = labels

    return file_lists

file_lists = get_file_list(parent_dir, channels, label_pattern)


# check if lenght of all file lists is the same
# if not, print a warning
lengths = [len(file_lists[channel]) for channel in file_lists.keys()]
if len(set(lengths)) > 1:
    print("Warning: The number of label files in the different channels is not the same.")
    print("Number of label files in each channel:", lengths)


# Create a list to store row data
csv_rows = []
 
for i in range(len(file_lists[target])):
    # get regionprops of shots
    target_props = regionprops(io.imread(file_lists[target][i]))
    # add props of other channels. Account for variable number of channels
    other_props = {}
    for channel in file_lists.keys():
        if channel != target:
            other_props[channel] = regionprops(io.imread(file_lists[channel][i]))


    # extract target file name without extension for CSV file name
    target_filename = os.path.splitext(os.path.basename(file_lists[target][i]))[0]

    # Loop through each region in target_props and check for centroids in other channels
    for target_prop in target_props:
        target_area = target_prop.area
        if 100 < target_area < 100000:
            target_bbox = target_prop.bbox
            target_min_row, target_min_col, target_max_row, target_max_col = target_bbox

            # Count the number of centroids in the corresponding target bbox
            centroid_count = {}
            for channel in other_props.keys():
                centroid_count[channel] = 0
                for other_prop in other_props[channel]:
                    centroid = other_prop.centroid
                    row, col = int(centroid[0]), int(centroid[1])
                    bbox = other_prop.bbox
                    min_row, min_col, max_row, max_col = bbox

                    if target_min_row <= row <= target_max_row and target_min_col <= col <= target_max_col:
                        centroid_count[channel] += 1

            # Append row data to the list
            csv_rows.append([target_filename, target_prop.label, target_prop.area,
                                *[centroid_count[channel] for channel in other_props.keys()] 
                                # the * unpacks the list of centroid_count values
                                ]) 

# Define the path for the combined CSV file
output_csv = os.path.join(parent_dir, 'colocalization_count.csv')

# Write the row data to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Filename', f'{target} Labels', 
                        f'{target} Area (sq px)',
                        *other_props.keys()])
    csvwriter.writerows(csv_rows)
            






