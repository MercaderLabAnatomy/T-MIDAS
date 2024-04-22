"""

This script validates automated segmentation results against manual segmentation results.

"""

import os
import argparse
from skimage.measure import regionprops as regionprops_cpu
from skimage.io import imread
import pandas as pd
from cucim.skimage.measure import regionprops as regionprops_gpu
import cupy as cp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the segmentation results.')
    parser.add_argument('--label_pattern', type=str, help='Label image suffix. Example: "_labels.tif"')
    parser.add_argument('--gt_pattern', type=str, help='Ground truth label image suffix. Example: "_ground_truth.tif"')
    return parser.parse_args()

args = parse_args()

input_folder = args.input
label_pattern = args.label_pattern
gt_pattern = args.gt_pattern

predictions = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(label_pattern)]
ground_truths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(gt_pattern)] 
predictions.sort()
ground_truths.sort()

results = []

def validation_cpu(ground_truth_file, prediction_file):
    gt_props = regionprops_cpu(imread(ground_truth_file))
    pred_props = regionprops_cpu(imread(prediction_file))
    gt_labels = [prop.label for prop in gt_props]
    pred_labels = [prop.label for prop in pred_props]
    gt_centroids = [prop.centroid for prop in gt_props]
    pred_bboxes = [prop.bbox for prop in pred_props]
    num_gt_inside_pred = 0
    num_gt_objects = len(gt_labels)
    num_pred_objects = len(pred_labels)
    sample = os.path.basename(ground_truth_file)[:-17]
    
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

def validation_gpu(ground_truth_file, prediction_file):
    gt_props = regionprops_gpu(cp.asarray(imread(ground_truth_file)))
    pred_props = regionprops_gpu(cp.asarray(imread(prediction_file)))
    gt_labels = [prop.label for prop in gt_props]
    pred_labels = [prop.label for prop in pred_props]
    gt_centroids = [prop.centroid for prop in gt_props]
    pred_bboxes = [prop.bbox for prop in pred_props]
    num_gt_inside_pred = 0
    num_gt_objects = len(gt_labels)
    num_pred_objects = len(pred_labels)
    sample = os.path.basename(ground_truth_file)[:-17]

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

for idx, i in enumerate(tqdm(ground_truth_files, total = len(ground_truth_files), desc="Processing images")):
    print(f"Processing {idx+1} of {len(ground_truth_files)}")

    if cp.cuda.is_available():
        validation_gpu(ground_truth_files[i], prediction_files[i])
    else:
        validation_cpu(ground_truth_files[i], prediction_files[i])

output_df = pd.DataFrame(results, columns=['sample', 'num_gt_objects', 'num_pred_objects', 'gt_centroids_inside_pred_bboxes'])
output_df.drop_duplicates(inplace=True)
output_df.to_csv(os.path.join(input_folder, 'validation_results.csv'), index=False)
