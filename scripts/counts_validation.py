"""

    This script validates automated segmentation results against manual segmentation results.
    
"""
    
import os
import argparse
from skimage.measure import regionprops
from skimage.io import imread
import pandas as pd


parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
parser.add_argument('--input', type=str, help='Path to the folder containing the segmentation results.')


args = parser.parse_args()
input_folder = args.input

#input_folder = "/home/marco/Pictures/test_validation"

prediction_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_labels.tif')]
ground_truth_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_ground_truth.tif')]
prediction_files.sort()
ground_truth_files.sort()


# check if ground truth centroids are inside the prediction ROIs
results = []
for i in range(len(ground_truth_files)):
    gt_props = regionprops(imread(ground_truth_files[i]))
    pred_props = regionprops(imread(prediction_files[i]))
    gt_labels = [prop.label for prop in gt_props]
    pred_labels = [prop.label for prop in pred_props]
    gt_centroids = [prop.centroid for prop in gt_props]
    pred_bboxes = [prop.bbox for prop in pred_props]
    num_gt_inside_pred = 0
    num_gt_objects = len(gt_labels)
    num_pred_objects = len(pred_labels)
    # filename without _ground_truth.tif and path
    sample = os.path.basename(ground_truth_files[i])[:-17]
    for j in range(len(gt_labels)):
        gt_label = gt_labels[j]
        gt_centroid = gt_centroids[j]

        for k in range(len(pred_labels)):
            pred_label = pred_labels[k]
            pred_bbox = pred_bboxes[k]
            if gt_label == pred_label:
                if pred_bbox[0] <= gt_centroid[0] <= pred_bbox[2] and pred_bbox[1] <= gt_centroid[1] <= pred_bbox[3]:
                    num_gt_inside_pred += 1
    results.append([sample, num_gt_objects, num_pred_objects, num_gt_inside_pred])
        
output_df = pd.DataFrame(results, columns=['sample', 'num_gt_objects', 'num_pred_objects', 'gt_centroids_inside_pred_bboxes'])
output_df = output_df.drop_duplicates()
output_df.to_csv(os.path.join(input_folder, 'validation_results.csv'), index=False)
