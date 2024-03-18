import os
import glob
import csv
from skimage import io
from skimage.measure import regionprops
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels',  nargs='+', type=str, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--add_intensity', type=str, help='Do you want to quantify average intensity of C2 in C1 ROI? (y/n)')
    parser.add_argument('--label_patterns', nargs='+', type=str, help='Label pattern for each channel. Example: "_labels.tif _labels.tif _labels.tif"')
    return parser.parse_args()


args = parse_arguments()

#label_pattern = '*_labels.tif'
parent_dir = args.input +'/'
channels = [c.upper() for c in args.channels]
label_patterns = [p for p in args.label_patterns]

# give error if channel names are not unique or smaller than 2
if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")



# def get_file_list(parent_dir, channels, label_pattern):
#     file_lists = {}  # Dictionary to store lists with channel names as keys

#     for channel in channels:
#         labels = sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern)))
#         file_lists[channel] = labels

#     return file_lists

# file_lists = get_file_list(parent_dir, channels, label_pattern)


def get_file_list(parent_dir, channels, label_patterns):
    file_lists = {}  # Dictionary to store lists with channel names as keys
    
    for channel, label_pattern in zip(channels, label_patterns):
        labels = sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern)))
        file_lists[channel] = labels
        
    return file_lists


file_lists = get_file_list(parent_dir, channels, label_patterns)


# check if length of all file lists is the same
# if not, print a warning
lengths = [len(file_lists[channel]) for channel in file_lists.keys()]
if len(set(lengths)) > 1:
    print("Warning: The number of label files in the different channels is not the same.")
    print("Number of label files in each channel:", lengths)


csv_rows = []
 


def coloc_3_channels(file_lists, channels, csv_rows):


    for i in range(len(file_lists[channels[0]])):

        # read images and get regionprops
        C0_img = io.imread(file_lists[channels[0]][i])
        C1_img = io.imread(file_lists[channels[1]][i])
        C2_img = io.imread(file_lists[channels[2]][i])

        C0_props = regionprops(C0_img)
        C1_props = regionprops(C1_img)
        C2_props = regionprops(C2_img)

        C0_filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]



        # Loop through each region in C0_props and check for C1 and C2 centroids
        for C0_prop in C0_props:
            C0_area = C0_prop.area
            if 100 < C0_area:# < 100000:
                C0_bbox = C0_prop.bbox
                C0_min_row, C0_min_col, C0_max_row, C0_max_col = C0_bbox

                # Check whether a C1 centroid is in the corresponding C0 bbox
                C1_centroid_in_C0_bbox = False

                for C1_prop in C1_props:
                    C1_centroid = C1_prop.centroid
                    C1_row, C1_col = int(C1_centroid[0]), int(C1_centroid[1])
                    C1_bbox = C1_prop.bbox
                    C1_min_row, C1_min_col, C1_max_row, C1_max_col = C1_bbox

                    if C0_min_row <= C1_row <= C0_max_row and C0_min_col <= C1_col <= C0_max_col:
                        C1_centroid_in_C0_bbox = True


                        # Check whether a C2 centroid is in the corresponding C1 bbox
                        C2_centroid_in_C1_bbox = False

                        for C2_prop in C2_props:
                            C2_centroid = C2_prop.centroid
                            C2_row, C2_col = int(C2_centroid[0]), int(C2_centroid[1])
                            if C1_min_row <= C2_row <= C1_max_row and C1_min_col <= C2_col <= C1_max_col:
                                C2_centroid_in_C1_bbox = True
                                break
                        # get average intensity of C2 in C1 ROI
                        mask = np.zeros_like(C1_img, dtype=bool)
                        mask[C1_prop.coords[:, 0], C1_prop.coords[:, 1]] = True
                        mean_intensity = np.mean(C2_img[mask])
                        if args.add_intensity == 'y':
                            csv_rows.append([C0_filename, C0_prop.label, C0_area, C1_centroid_in_C0_bbox, C2_centroid_in_C1_bbox, mean_intensity])
                        else:
                            csv_rows.append([C0_filename, C0_prop.label, C0_area, C1_centroid_in_C0_bbox, C2_centroid_in_C1_bbox])


def coloc_2_channels(file_lists, channels, csv_rows):


    for i in range(len(file_lists[channels[0]])):

        # read images and get regionprops
        C0_img = io.imread(file_lists[channels[0]][i])
        C1_img = io.imread(file_lists[channels[1]][i])

        C0_props = regionprops(C0_img)
        C1_props = regionprops(C1_img)

        C0_filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]



        # Loop through each region in C0_props and check for C1 and C2 centroids
        for C0_prop in C0_props:
            C0_area = C0_prop.area
            if 100 < C0_area:# < 100000:
                C0_bbox = C0_prop.bbox
                C0_min_row, C0_min_col, C0_max_row, C0_max_col = C0_bbox

                # Check whether a C1 centroid is in the corresponding C0 bbox
                C1_centroid_in_C0_bbox = False

                for C1_prop in C1_props:
                    C1_centroid = C1_prop.centroid
                    C1_row, C1_col = int(C1_centroid[0]), int(C1_centroid[1])
                    C1_bbox = C1_prop.bbox
                    #C1_min_row, C1_min_col, C1_max_row, C1_max_col = C1_bbox

                    if C0_min_row <= C1_row <= C0_max_row and C0_min_col <= C1_col <= C0_max_col:
                        C1_centroid_in_C0_bbox = True
                        break
                # Get average intensity of C1 in C0 ROI
                    mask = np.zeros_like(C0_img, dtype=bool)
                    mask[C0_prop.coords[:, 0], C0_prop.coords[:, 1]] = True
                    mean_intensity = np.mean(C1_img[mask])

                    




                if args.add_intensity == 'y':
                    csv_rows.append([C0_filename, C0_prop.label, C0_area, C1_centroid_in_C0_bbox, mean_intensity])
                else:
                    csv_rows.append([C0_filename, C0_prop.label, C0_area, C1_centroid_in_C0_bbox])


def create_csv(file_lists, channels, csv_rows):
    if len(channels) == 3:
        coloc_3_channels(file_lists, channels, csv_rows)
    elif len(channels) == 2:
        coloc_2_channels(file_lists, channels, csv_rows)
    else:
        raise ValueError("This script only supports 2 or 3 channels.")
    
create_csv(file_lists, channels, csv_rows)





output_csv = os.path.join(parent_dir, 'colocalization.csv')

    
if len(channels) == 3:
    if args.add_intensity == 'y':
        header = ['Filename', 
                  f'{channels[0]}_label', f'{channels[0]}_area', 
                  f'{channels[1]}_centroid_in_{channels[0]}_bbox', 
                  f'{channels[2]}_centroid_in_{channels[1]}_bbox', 
                  f'{channels[2]}_mean_intensity']
    else:
        header = ['Filename', 
                  f'{channels[0]}_label', f'{channels[0]}_area', 
                  f'{channels[1]}_centroid_in_{channels[0]}_bbox', 
                  f'{channels[2]}_centroid_in_{channels[1]}_bbox']



with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_rows)


print("Colocalization data saved to", output_csv)
print("Done.")