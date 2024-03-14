# this script takes two folders with TIF files as input: one with blob channel and one with ROI channel
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
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
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
parser = argparse.ArgumentParser(description='Get blob inside ROI')
parser.add_argument('--input_folder', type=str, help='Path to folder containing blob and ROI label image subfolders')
# ask for names of the subfolders
parser.add_argument('--blob_folder', type=str, help='Name of the folder containing blob label images')
parser.add_argument('--ROI_folder', type=str, help='Name of the folder containing ROI label images')
#parser.add_argument("--pixel_resolution", type=float, required=True, help="Pixel resolution of the images in um/px.")
args = parser.parse_args()



blob_folder = os.path.join(args.input_folder, blob_folder)
ROI_folder = os.path.join(args.input_folder, ROI_folder)


# take parent folder as directory for output folder
output_folder_dir = os.path.dirname(blob_folder)
# Create output folder if it doesn't exist
output_folder = os.path.join(output_folder_dir, 'output')
os.makedirs(output_folder, exist_ok=True)




features = 'area,mean_max_distance_to_centroid_ratio'

cl_filename = os.path.join(os.environ['TMIDAS_PATH'], "models/ObjectClassifier_blob.cl")
# classifier_labels = imread("/opt/models/blob_in_ROI/20230821_ehd2_laser_abl_02_HT_ab_Position002_cropped_blob_in_ROI_labels.tif")
# # classifier_image = imread("models/blob_in_ROI/20230821_ehd2_laser_abl_02_HT_ab_Position002_cropped_blob_intensities.tif")
# classifier_annotation = imread("models/blob_in_ROI/20230821_ehd2_laser_abl_02_HT_ab_Position002_blob_annotations.tif")



# Create an object classifier
# apoc.erase_classifier(cl_filename) # delete it if it was existing before
classifier = apoc.ObjectClassifier(cl_filename)
# # train it
# classifier.train(features, labels=classifier_labels, sparse_annotation=classifier_annotation)
# classifier.feature_importances()

results_list = []

# Iterate over files in blob_folder
for blob_image_filename in os.listdir(blob_folder):
    if blob_image_filename.endswith("blob_labels.tif"):
        # Load blob_image

        blob_labels = load_image(os.path.join(blob_folder, blob_image_filename))
        #blob_intensities = load_image(os.path.join(blob_folder, blob_image_filename.replace('blob_labels', 'blob_intensities')))
        
        # replace 'blob' with 'ROI' in ROI_image_filenames
        ROI_image_filename = blob_image_filename.replace('blob_labels', 'ROI_labels')

        ROI_labels = load_image(os.path.join(ROI_folder, ROI_image_filename))
        volume = get_volume(ROI_labels)
        #image = load_image("/home/marco/Pictures/ImagesMarwa/20230821_ehd2_laser_abl_02_HM_ab_Position005_cropped_ROI_labels.tif")

        
        # Remove all blob labels outside of ROI
        blob_labels_in_ROI = cle.binary_intersection(ROI_labels, blob_labels)


        blob_labels_in_ROI = cle.connected_components_labeling_box(blob_labels_in_ROI)
        blob_labels_in_ROI = cle.exclude_small_labels(blob_labels_in_ROI, None, 500)
        classifier = apoc.ObjectClassifier(cl_filename)
        blob_classes = classifier.predict(blob_labels_in_ROI)        
        blob_labels_in_ROI_filename = blob_image_filename.replace('blob', 'blob_in_ROI')
        common_part = blob_labels_in_ROI_filename.split('_blob')[0]

        for value in np.unique(cle.pull(blob_classes)[cle.pull(blob_classes) != 0]):
            sub_array = np.zeros_like(cle.pull(blob_classes) )
            sub_array[cle.pull(blob_classes) == value] = value
            blob_class = cle.binary_and(cle.push(sub_array), blob_labels_in_ROI)
            blob_class = cle.connected_components_labeling_box(blob_class)
            blob_count = cle.maximum_of_all_pixels(blob_class)
            print(f"Sample: {common_part}, Count: {int(blob_count)}, Class: {int(value)}, Volume (um3): {int(volume)}")    
            # Add the result to the DataFrame
            result = {"Sample": common_part, "Count": int(blob_count), "Class": int(value), "Volume (um3)": int(volume)}    
            results_list.append(result)
        
        # num_blob_labels_in_ROI = cle.maximum_of_all_pixels(blob_labels_in_ROI)
        # Save labels
        # replace 'blob' with 'blob_in_ROI' in blob_image_filenames

        blob_classes_filename = blob_image_filename.replace('blob', 'blob_classes')             
        napari_io.imsave(os.path.join(output_folder, blob_labels_in_ROI_filename), blob_labels_in_ROI)
        napari_io.imsave(os.path.join(output_folder, blob_classes_filename), blob_classes)
        # get common part of filename

        print(f"Sample: {common_part}, Count: {int(blob_count)}, Class: {int(value)}, Volume (um3): {int(volume)}")  
        # Add the result to the DataFrame
        result = {"Sample": common_part, "Count": int(blob_count), "Class": int(value), "Volume (um3)": int(volume)}    
        results_list.append(result)

results = pd.DataFrame(results_list)

# Save CSV
output_csv = os.path.join(output_folder, "all_sample_counts.csv")
results.to_csv(output_csv, index=False)
