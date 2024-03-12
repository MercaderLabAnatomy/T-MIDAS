import os
import glob
import csv
from skimage import io
from skimage.measure import regionprops
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Count proliferating FITC+ cells.')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels',  nargs='+', type=str, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
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




#channels = ["FITC", "TRITC", "DAPI"]
# parent_dir = "/media/geffjoldblum/DATA/images_joao/20240205_ntn1a_col1a2_IB4"
# label_pattern = "*_labels.tif"

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
 

for i in range(len(file_lists[channels[0]])):
    """
    This loop iterates through the label images of the first channel 
    and 
    - counts the number of regions in the second channel 
    that are inside the bounding box of the first channel, and
    - counts the number of regions in the third channel that
    are inside the bounding box of the first and second channels.
    """

    props = {}
    for channel in channels:
        img = io.imread(file_lists[channel][i])
        props[channel] = regionprops(img)
    
    
    filename = os.path.splitext(os.path.basename(file_lists[channels[0]][i]))[0]
    
    # Initialize counters
    count_in_bbox_1 = 0
    count_in_bbox_1_and_bbox_2 = 0
    
    # loop through regions in the second channel
    for region in props[channels[1]]:
        centroid = region.centroid
        area = region.area
        label = region.label
        if 100 < area:
        
            # check if the centroid is inside the bbox of the first channel
            for region in props[channels[0]]:
                minr, minc, maxr, maxc = region.bbox
                if minr < centroid[0] < maxr and minc < centroid[1] < maxc:
                    count_in_bbox_1 += 1

                for region in props[channels[2]]:
                    minr, minc, maxr, maxc = region.bbox
                    if minr < centroid[0] < maxr and minc < centroid[1] < maxc:
                        count_in_bbox_1_and_bbox_2 += 1
                        break
            # append row data to the list
            csv_rows.append([filename, label, area, count_in_bbox_1, count_in_bbox_1_and_bbox_2])  
            
# define path to save the csv file
csv_file = os.path.join(parent_dir, 'colocalization_counts.csv')


# write row data to a csv file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", f"{channels[0]} ROI id", f"{channels[0]} ROI area (sq px)", 
                     f"Number of {channels[1]} ROIs in {channels[0]} ROIs", f"Number of {channels[2]} ROIs in {channels[0]} and {channels[1]} ROIs" ])
    writer.writerows(csv_rows)
    