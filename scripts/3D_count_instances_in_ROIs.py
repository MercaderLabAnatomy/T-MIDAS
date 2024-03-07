# this script takes two folders with TIF files as input: one with nuclei channel and one with tissue channel
# it requires to be run in the napari-apoc conda environment




import os
import numpy as np
import argparse
import pyclesperanto_prototype as cle
from napari.utils import io as napari_io
from skimage.io import imread, imsave
import pandas as pd
import apoc
import napari_segment_blobs_and_things_with_membranes as nsbatwm 
import SimpleITK as sitk
from skimage.measure import regionprops
import pyclesperanto_prototype as cle



def load_image(image_path):
    try:
        return imread(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
pixel_width, pixel_height, pixel_depth = 0.2840909, 0.2840909, 1.5001314  
    
# get volume in um^3
def get_volume(ROI):
    volume = cle.sum_of_all_pixels(ROI)
    volume_um3 = volume * (pixel_width * pixel_height * pixel_depth)
    return volume_um3

# Argument parsing
parser = argparse.ArgumentParser(description='Get nuclei inside tissue')
parser.add_argument('--input_folder', type=str, help='Path to folder containing nuclei and tissue label image subfolders')
# ask for names of the subfolders
parser.add_argument('--nuclei_folder', type=str, help='Name of the folder containing nuclei label images')
parser.add_argument('--tissue_folder', type=str, help='Name of the folder containing tissue label images')
#parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
args = parser.parse_args()



nuclei_folder = os.path.join(args.input_folder, nuclei_folder)
tissue_folder = os.path.join(args.input_folder, tissue_folder)


# take parent folder as directory for output folder
output_folder_dir = os.path.dirname(nuclei_folder)
# Create output folder if it doesn't exist
output_folder = os.path.join(output_folder_dir, 'output')
os.makedirs(output_folder, exist_ok=True)




features = 'area,mean_max_distance_to_centroid_ratio'
cl_filename = "/mnt/disk2/Marco/nuclei_in_tissue/ObjectClassifier_nuclei.cl"
# classifier_labels = imread("/opt/models/nuclei_in_tissue/20230821_ehd2_laser_abl_02_HT_ab_Position002_cropped_nuclei_in_tissue_labels.tif")
# # classifier_image = imread("models/nuclei_in_tissue/20230821_ehd2_laser_abl_02_HT_ab_Position002_cropped_nuclei_intensities.tif")
# classifier_annotation = imread("models/nuclei_in_tissue/20230821_ehd2_laser_abl_02_HT_ab_Position002_nuclei_annotations.tif")



# cle.imshow(classifier_labels, labels=True)

# Create an object classifier
# apoc.erase_classifier(cl_filename) # delete it if it was existing before
classifier = apoc.ObjectClassifier(cl_filename)
# # train it
# classifier.train(features, labels=classifier_labels, sparse_annotation=classifier_annotation)
# classifier.feature_importances()

results_list = []

# Iterate over files in nuclei_folder
for nuclei_image_filename in os.listdir(nuclei_folder):
    if nuclei_image_filename.endswith("nuclei_labels.tif"):
        # Load nuclei_image

        nuclei_labels = load_image(os.path.join(nuclei_folder, nuclei_image_filename))
        nuclei_intensities = load_image(os.path.join(nuclei_folder, nuclei_image_filename.replace('nuclei_labels', 'nuclei_intensities')))
        
        # replace 'nuclei' with 'tissue' in tissue_image_filenames
        tissue_image_filename = nuclei_image_filename.replace('nuclei_labels', 'tissue_labels')

        tissue_labels = load_image(os.path.join(tissue_folder, tissue_image_filename))
        volume = get_volume(tissue_labels)
        #image = load_image("/home/marco/Pictures/ImagesMarwa/20230821_ehd2_laser_abl_02_HM_ab_Position005_cropped_tissue_labels.tif")

        
        # Remove all nuclei labels outside of tissue
        nuclei_labels_in_tissue = cle.binary_intersection(tissue_labels, nuclei_labels)


        nuclei_labels_in_tissue = cle.connected_components_labeling_box(nuclei_labels_in_tissue)
        nuclei_labels_in_tissue = cle.exclude_small_labels(nuclei_labels_in_tissue, None, 500)
        classifier = apoc.ObjectClassifier(cl_filename)
        nuclei_classes = classifier.predict(nuclei_labels_in_tissue)        
        nuclei_labels_in_tissue_filename = nuclei_image_filename.replace('nuclei', 'nuclei_in_tissue')
        common_part = nuclei_labels_in_tissue_filename.split('_nuclei')[0]

        for value in np.unique(cle.pull(nuclei_classes)[cle.pull(nuclei_classes) != 0]):
            sub_array = np.zeros_like(cle.pull(nuclei_classes) )
            sub_array[cle.pull(nuclei_classes) == value] = value
            nuclei_class = cle.binary_and(cle.push(sub_array), nuclei_labels_in_tissue)
            nuclei_class = cle.connected_components_labeling_box(nuclei_class)
            nuclei_count = cle.maximum_of_all_pixels(nuclei_class)
            print(f"Sample: {common_part}, Count: {int(nuclei_count)}, Class: {int(value)}, Volume (um3): {int(volume)}")    
            # Add the result to the DataFrame
            result = {"Sample": common_part, "Count": int(nuclei_count), "Class": int(value), "Volume (um3)": int(volume)}    
            results_list.append(result)
        
        # num_nuclei_labels_in_tissue = cle.maximum_of_all_pixels(nuclei_labels_in_tissue)
        # Save labels
        # replace 'nuclei' with 'nuclei_in_tissue' in nuclei_image_filenames

        nuclei_classes_filename = nuclei_image_filename.replace('nuclei', 'nuclei_classes')             
        napari_io.imsave(os.path.join(output_folder, nuclei_labels_in_tissue_filename), nuclei_labels_in_tissue)
        napari_io.imsave(os.path.join(output_folder, nuclei_classes_filename), nuclei_classes)
        # get common part of filename

        print(f"Sample: {common_part}, Count: {int(nuclei_count)}, Class: {int(value)}, Volume (um3): {int(volume)}")  
        # Add the result to the DataFrame
        result = {"Sample": common_part, "Count": int(nuclei_count), "Class": int(value), "Volume (um3)": int(volume)}    
        results_list.append(result)

results = pd.DataFrame(results_list)

# Save CSV
output_csv = os.path.join(output_folder, "all_sample_counts.csv")
results.to_csv(output_csv, index=False)
