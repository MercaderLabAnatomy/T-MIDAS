import os
import numpy as np
import argparse
import pyclesperanto_prototype as cle
from napari.utils import io as napari_io
from skimage.io import imread
import pandas as pd
import apoc
import pyclesperanto_prototype as cle
from tqdm import tqdm

def load_image(image_path):
    try:
        return imread(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def get_volume(ROI, pixel_width, pixel_height, pixel_depth):
    volume = cle.sum_of_all_pixels(ROI)
    volume_um3 = volume * (pixel_width * pixel_height * pixel_depth)
    return volume_um3

def parse_args():
    parser = argparse.ArgumentParser(description='Get blob inside ROI')
    parser.add_argument('--input_folder', type=str, help='Path to folder containing blob and ROI label image subfolders')
    parser.add_argument('--blob_folder', type=str, help='Name of the folder containing blob label images')
    parser.add_argument('--ROI_folder', type=str, help='Name of the folder containing ROI label images')
    parser.add_argument('--pixel_width', type=float, help='Pixel width in micrometers')
    parser.add_argument('--pixel_height', type=float, help='Pixel height in micrometers')
    parser.add_argument('--pixel_depth', type=float, help='Pixel depth in micrometers')
    return parser.parse_args()

args = parse_args()

blob_folder = os.path.join(args.input_folder, args.blob_folder)
ROI_folder = os.path.join(args.input_folder, args.ROI_folder)

output_folder_dir = os.path.dirname(blob_folder)
output_folder = os.path.join(output_folder_dir, 'output')
os.makedirs(output_folder, exist_ok=True)

features = 'area,mean_max_distance_to_centroid_ratio'

cl_filename = os.path.join(os.environ['TMIDAS_PATH'], "models/ObjectClassifier_blob.cl")

classifier = apoc.ObjectClassifier(cl_filename)

results_list = []

for blob_image_filename in tqdm(os.listdir(blob_folder), total=len(os.listdir(blob_folder)), desc="Processing images"):
    if blob_image_filename.endswith("blob_labels.tif"):
        blob_labels = load_image(os.path.join(blob_folder, blob_image_filename))
        
        ROI_image_filename = blob_image_filename.replace('blob_labels', 'ROI_labels')
        ROI_labels = load_image(os.path.join(ROI_folder, ROI_image_filename))
        volume = get_volume(ROI_labels, args.pixel_width, args.pixel_height, args.pixel_depth)

        blob_labels_in_ROI = cle.binary_intersection(ROI_labels, blob_labels)
        blob_labels_in_ROI = cle.connected_components_labeling_box(blob_labels_in_ROI)
        blob_labels_in_ROI = cle.exclude_small_labels(blob_labels_in_ROI, None, 500)
        
        blob_classes = classifier.predict(blob_labels_in_ROI)        
        blob_labels_in_ROI_filename = blob_image_filename.replace('blob', 'blob_in_ROI')
        common_part = blob_labels_in_ROI_filename.split('_blob')[0]

        for value in np.unique(cle.pull(blob_classes)[cle.pull(blob_classes) != 0]):
            sub_array = np.zeros_like(cle.pull(blob_classes))
            sub_array[cle.pull(blob_classes) == value] = value
            blob_class = cle.binary_and(cle.push(sub_array), blob_labels_in_ROI)
            blob_class = cle.connected_components_labeling_box(blob_class)
            blob_count = cle.maximum_of_all_pixels(blob_class)
            print(f"Sample: {common_part}, Count: {int(blob_count)}, Class: {int(value)}, Volume (um3): {int(volume)}")    
            result = {"Sample": common_part, "Count": int(blob_count), "Class": int(value), "Volume (um3)": int(volume)}    
            results_list.append(result)
        
        blob_classes_filename = blob_image_filename.replace('blob', 'blob_classes')             
        napari_io.imsave(os.path.join(output_folder, blob_labels_in_ROI_filename), blob_labels_in_ROI)
        napari_io.imsave(os.path.join(output_folder, blob_classes_filename), blob_classes)

results = pd.DataFrame(results_list)

output_csv = os.path.join(output_folder, "all_sample_counts.csv")
results.to_csv(output_csv, index=False)
