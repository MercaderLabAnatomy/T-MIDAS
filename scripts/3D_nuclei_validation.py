import os
import argparse
from sklearn import metrics
from skimage.io import imread
import pandas as pd
import numpy as np
import pyclesperanto_prototype as cle
from the_segmentation_game import metrics as metrics_game


parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
parser.add_argument('--input', type=str, help='Path to the folder containing the segmentation results.')
# semantic or instance segmentation
parser.add_argument('--type', type=str, help='Type of segmentation: semantic or instance.')


args = parser.parse_args()
input_folder = args.input
segmentation_type = args.type


# input_folder = "/home/marco/Pictures/test_validation/3D_nuclei"
# segmentation_type = 'semantic'


 
predictions = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_labels.tif')]
ground_truths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_ground_truth.tif')] 
predictions.sort()
ground_truths.sort()

#threshold = 1
 
results = []
jaccard_type = ''

for i in range(len(predictions)):
    prediction = imread(predictions[i])
    ground_truth = imread(ground_truths[i])
    if segmentation_type == 's':
        jaccard_type = 'jaccard_index_binary'
        jaccard_index = metrics_game.jaccard_index_binary(ground_truth, prediction)
    elif segmentation_type == 'm':
        jaccard_type = 'jaccard_index_sparse'
        jaccard_index = metrics_game.jaccard_index_sparse(ground_truth, prediction)
    else:
        raise ValueError('Segmentation type not recognized.')
    
    
    # prediction_binary = prediction >= threshold
    # ground_truth_binary = ground_truth >= threshold
    
    # gt_1d = np.ravel(ground_truth_binary)
    # pred_1d = np.ravel(prediction_binary)
    # # convert boolean to int
    # gt_1d = gt_1d.astype(int)
    # pred_1d = pred_1d.astype(int)
    
    # gt_1d_mc = np.ravel(ground_truth)
    # pred_1d_mc = np.ravel(prediction)
    
    # # print(np.unique(gt_1d_mc))
    # # print(np.unique(pred_1d_mc))
    
    # accuracy = metrics.accuracy_score(gt_1d, pred_1d)
    # precision = metrics.precision_score(gt_1d, pred_1d)
    # recall = metrics.recall_score(gt_1d, pred_1d)
    # f1 = metrics.f1_score(gt_1d, pred_1d)
    # jib = metrics.jaccard_score(gt_1d, pred_1d, average='binary') # binary classification
    #jis = metrics.jaccard_score(gt_1d_mc, pred_1d_mc, average='samples') # sparse multilabel classification
    
    filename = os.path.basename(predictions[i])[:-11]

    results.append([filename, jaccard_index])
 

# results = [[os.path.basename(predictions[i])[:-11],
#             metrics_game.jaccard_index_binary(imread(ground_truths[i]), imread(predictions[i])),
#             metrics_game.jaccard_index_sparse(imread(ground_truths[i]), imread(predictions[i])),
#             metrics.accuracy_score(np.ravel(imread(ground_truths[i]) >= threshold), np.ravel(imread(predictions[i]) >= threshold)),
#             metrics.precision_score(np.ravel(imread(ground_truths[i]) >= threshold), np.ravel(imread(predictions[i]) >= threshold)),
#             metrics.recall_score(np.ravel(imread(ground_truths[i]) >= threshold), np.ravel(imread(predictions[i]) >= threshold)),
#             metrics.f1_score(np.ravel(imread(ground_truths[i]) >= threshold), np.ravel(imread(predictions[i]) >= threshold))]
#            for i in range(len(predictions))]


  
output_df = pd.DataFrame(results, columns=['filename', jaccard_type])
output_df.to_csv(os.path.join(input_folder, 'results.csv'), index=False)


