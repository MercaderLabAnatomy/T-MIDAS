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
    return parser.parse_args()


args = parse_arguments()

label_pattern = '*_labels.tif'
parent_dir = args.input +'/'
channels = [c.upper() for c in args.channels]

# give error if channel names are not unique or smaller than 2
if len(set(channels)) < len(channels) or len(channels) < 2:
    raise ValueError("Channel names must be unique and at least two channels must be provided.")



def get_file_list(parent_dir, channels, label_pattern):
    file_lists = {}  # Dictionary to store lists with channel names as keys

    for channel in channels:
        labels = sorted(glob.glob(os.path.join(parent_dir, channel + '/', label_pattern)))
        file_lists[channel] = labels

    return file_lists

file_lists = get_file_list(parent_dir, channels, label_pattern)


# check if length of all file lists is the same
# if not, print a warning
lengths = [len(file_lists[channel]) for channel in file_lists.keys()]
if len(set(lengths)) > 1:
    print("Warning: The number of label files in the different channels is not the same.")
    print("Number of label files in each channel:", lengths)


csv_rows = []
 

# for i in range(len(file_lists[channels[0]])):
#     # Extract filename without extension
#     filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]

#     # Initialize counters
#     rois1_count_in_bbox_0 = 0
#     rois2_count_in_bbox_0_and_bbox_1 = 0
#     rois2_count_in_bbox_0 = 0

#     props = {}
#     for channel in channels:
#         img = io.imread(file_lists[channel][i])
#         props[channel] = regionprops(img)

#     # Loop through regions in the zeroth channel
#     for ROIs0 in props[channels[0]]:
#         # count centroids of first channel in bbox of zeroth channel
#         for ROIs1 in props[channels[1]]:
#             if ROIs0.bbox[0] <= ROIs1.centroid[0] <= ROIs0.bbox[2] and ROIs0.bbox[1] <= ROIs1.centroid[1] <= ROIs0.bbox[3]:
#                 rois1_count_in_bbox_0 += 1
#                 if len(channels) == 3:
#                     for ROIs2 in props[channels[2]]:
#                         if ROIs1.bbox[0] <= ROIs2.centroid[0] <= ROIs1.bbox[2] and ROIs1.bbox[1] <= ROIs2.centroid[1] <= ROIs1.bbox[3]:
#                             rois2_count_in_bbox_0_and_bbox_1 += 1
#                         elif (ROIs0.bbox[0] <= ROIs2.centroid[0] <= ROIs0.bbox[2] and ROIs0.bbox[1] <= ROIs2.centroid[1] <= ROIs0.bbox[3]):
#                             rois2_count_in_bbox_0 += 1

#                     csv_rows.append([filename, ROIs0.label, ROIs0.area, rois1_count_in_bbox_0, 
#                                      rois2_count_in_bbox_0 , rois2_count_in_bbox_0_and_bbox_1])
#                 elif len(channels) == 2:
#                     csv_rows.append([filename, ROIs0.label, ROIs0.area, rois1_count_in_bbox_0])
#                 else:
#                     raise ValueError("Number of channels must be 2 or 3.")



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


output_csv = os.path.join(parent_dir, 'colocalization.csv')


if args.add_intensity == 'y':
    header = ['Filename', 'C0_label', 'C0_area', 'C1_centroid_in_C0_bbox', 'C2_centroid_in_C1_bbox', 'C2_mean_intensity']
else:
    header = ['Filename', 'C0_label', 'C0_area', 'C1_centroid_in_C0_bbox', 'C2_centroid_in_C1_bbox']

with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_rows)

print("Colocalization data saved to", output_csv)
print("Done.")