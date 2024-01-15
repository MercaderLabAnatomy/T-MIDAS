import os
import glob
import csv
from skimage import io
from skimage.measure import regionprops
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Count proliferating FITC+ cells.')
    parser.add_argument('--input', type=str, help='Path to the parent folder of the channel folders with segmented tiles.')
    # parser.add_argument('--num_tiles', type=int, help='Enter the number of tiles to sample.')
    return parser.parse_args()


args = parse_arguments()


# Define the naming pattern for label and intensity images
label_pattern = '*_labels.tif'
# label_pattern_2 = '*_cp_masks.tif' # cellpose

parent_dir = args.input +'/'


# Get the list of label files
FITC_labels = sorted(glob.glob(os.path.join(parent_dir + 'FITC/', label_pattern)))
TRITC_labels = sorted(glob.glob(os.path.join(parent_dir + 'TRITC/', label_pattern)))
DAPI_labels = sorted(glob.glob(os.path.join(parent_dir + 'DAPI/', label_pattern)))

FITC_intensities = [filename.replace("_labels", "") if "_labels" in filename else filename for filename in FITC_labels]
TRITC_intensities = [filename.replace("_labels", "") if "_labels" in filename else filename for filename in TRITC_labels]
DAPI_intensities = [filename.replace("_labels", "") if "_labels" in filename else filename for filename in DAPI_labels]


# # Create a new directory called "output" in the parent directory
# output_dir = os.path.join(parent_dir, "output")
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
    
# Create a list to store row data
csv_rows = []
 
# Loop through each element in FITC_labels and FITC_intensities and apply regionprops
for i in range(len(FITC_labels)):
    FITC_props = regionprops(io.imread(FITC_labels[i]), io.imread(FITC_intensities[i]))
    TRITC_props = regionprops(io.imread(TRITC_labels[i]), io.imread(TRITC_intensities[i]))
    DAPI_props = regionprops(io.imread(DAPI_labels[i]), io.imread(DAPI_intensities[i]))

    # Extract FITC file name without extension for CSV file name
    FITC_filename = os.path.splitext(os.path.basename(FITC_labels[i]))[0]
    # output_csv = os.path.join(output_dir, '{}_output.csv'.format(FITC_filename))

 

    # Loop through each region in FITC_props and check for DAPI and TRITC centroids
    for FITC_prop in FITC_props:
        FITC_area = FITC_prop.area
        FITC_min_intensity = FITC_prop.intensity_min
        if 100 < FITC_area < 100000 and FITC_min_intensity > 0:
            FITC_bbox = FITC_prop.bbox
            FITC_min_row, FITC_min_col, FITC_max_row, FITC_max_col = FITC_bbox

            # Check whether a DAPI centroid is in the corresponding FITC bbox
            DAPI_centroid_in_FITC_bbox = False

            for DAPI_prop in DAPI_props:
                DAPI_centroid = DAPI_prop.centroid
                DAPI_row, DAPI_col = int(DAPI_centroid[0]), int(DAPI_centroid[1])
                DAPI_bbox = DAPI_prop.bbox
                DAPI_min_row, DAPI_min_col, DAPI_max_row, DAPI_max_col = DAPI_bbox

                if FITC_min_row <= DAPI_row <= FITC_max_row and FITC_min_col <= DAPI_col <= FITC_max_col:
                    DAPI_centroid_in_FITC_bbox = True
                    

                    # Check whether a TRITC centroid is in the corresponding DAPI bbox
                    TRITC_centroid_in_DAPI_bbox = False

                    for TRITC_prop in TRITC_props:
                        TRITC_centroid = TRITC_prop.centroid
                        TRITC_row, TRITC_col = int(TRITC_centroid[0]), int(TRITC_centroid[1])



                        if DAPI_min_row <= TRITC_row <= DAPI_max_row and DAPI_min_col <= TRITC_col <= DAPI_max_col:
                            TRITC_centroid_in_DAPI_bbox = True
                            break

                    # Append row data to the list
                    csv_rows.append([FITC_filename, FITC_prop.label, FITC_prop.area * (0.23**2), FITC_prop.mean_intensity, 
                                     FITC_prop.major_axis_length* 0.23, FITC_prop.minor_axis_length* 0.23,
                                     FITC_prop.eccentricity,
                                     DAPI_centroid_in_FITC_bbox,TRITC_centroid_in_DAPI_bbox])

output_csv = os.path.join(parent_dir, 'regionprops.csv')  # Path for the combined CSV file
              
# Write the row data to the CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label', 'area_um2', 'mean_intensity', 'major_axis_length_um2','minor_axis_length_um2','eccentricity', 'DAPI_centroid_in_FITC_bbox', 'TRITC_centroid_in_DAPI_bbox'])
    writer.writerows(csv_rows)
print("\n"+ str(output_csv)+" created.")