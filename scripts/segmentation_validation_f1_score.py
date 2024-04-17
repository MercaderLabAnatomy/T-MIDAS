import os
import argparse
import cupy as cp
from skimage.io import imread
import pandas as pd
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
parser.add_argument('--input', type=str, help='Path to the folder containing the segmentation results.')
parser.add_argument('--label_pattern', type=str, help='Label image suffix. Example: "_labels.tif"')
parser.add_argument('--gt_pattern', type=str, help='Ground truth label image suffix. Example: "_ground_truth.tif"')
args = parser.parse_args()

input_folder = args.input
label_pattern = args.label_pattern
gt_pattern = args.gt_pattern

predictions = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(label_pattern)]
ground_truths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(gt_pattern)] 
predictions.sort()
ground_truths.sort()

# check if the number of files is the same
if len(predictions) != len(ground_truths):
    raise ValueError('Number of files does not match.')



 
results = []


for i in range(len(predictions)):
    prediction = imread(predictions[i])
    ground_truth = imread(ground_truths[i])
    f1 = f1_score(cp.ravel(ground_truth), cp.ravel(prediction), average='micro') 

    filename = os.path.basename(predictions[i])[:-11]

    results.append([filename, f1])
   
output_df = pd.DataFrame(results, columns=['filename', 'global_f1_score'])
output_df.to_csv(os.path.join(input_folder, 'results.csv'), index=False)


