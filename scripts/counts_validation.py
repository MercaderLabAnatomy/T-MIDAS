import os
import argparse
from skimage.measure import regionprops as regionprops_cpu
from skimage.io import imread
import pandas as pd
from tqdm import tqdm
import sys
from cucim.skimage.measure import regionprops as regionprops_gpu
import cupy as cp

def parse_args():
    parser = argparse.ArgumentParser(description='Validate segmentation results against manual segmentation results.')
    parser.add_argument('--input', type=str, required=True, help='Path to the folder containing the segmentation results.')
    parser.add_argument('--label_pattern', type=str, required=True, help='Label image suffix. Example: "_labels.tif"')
    parser.add_argument('--gt_pattern', type=str, required=True, help='Ground truth label image suffix. Example: "_ground_truth.tif"')
    return parser.parse_args()

def validate_segmentation(ground_truth_file, prediction_file, use_gpu):
    try:
        gt_image = imread(ground_truth_file)
        pred_image = imread(prediction_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return None
    except Exception as e:
        print(f"Error reading image files: {e}")
        return None

    try:
        if use_gpu:
            gt_props = regionprops_gpu(cp.asarray(gt_image))
            pred_props = regionprops_gpu(cp.asarray(pred_image))
        else:
            gt_props = regionprops_cpu(gt_image)
            pred_props = regionprops_cpu(pred_image)
    except Exception as e:
        print(f"Error during region properties computation: {e}")
        return None

    gt_centroids = [prop.centroid for prop in gt_props]
    pred_bboxes = [prop.bbox for prop in pred_props]

    num_gt_inside_pred = 0
    num_gt_objects = len(gt_props)
    num_pred_objects = len(pred_props)
    sample_name = os.path.basename(ground_truth_file).rsplit('_', 1)[0]

    for gt_centroid in gt_centroids:
        for pred_bbox in pred_bboxes:
            if (pred_bbox[0] <= gt_centroid[0] <= pred_bbox[2] and 
                pred_bbox[1] <= gt_centroid[1] <= pred_bbox[3]):
                num_gt_inside_pred += 1
                break

    return [sample_name, num_gt_objects, num_pred_objects, num_gt_inside_pred]

def main():
    args = parse_args()

    input_folder = args.input
    label_pattern = args.label_pattern
    gt_pattern = args.gt_pattern

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    predictions = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(label_pattern)])
    ground_truths = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(gt_pattern)])

    if len(predictions) != len(ground_truths):
        print(f"Error: Number of prediction files ({len(predictions)}) does not match number of ground truth files ({len(ground_truths)}).")
        sys.exit(1)

    if not predictions:
        print(f"Error: No files found matching the patterns '{label_pattern}' and '{gt_pattern}' in the input folder.")
        sys.exit(1)

    results = []
    use_gpu = cp.cuda.is_available()

    for gt_file, pred_file in tqdm(zip(ground_truths, predictions), total=len(ground_truths), desc="Processing images"):
        result = validate_segmentation(gt_file, pred_file, use_gpu)
        if result:
            results.append(result)

    if not results:
        print("Error: No valid results were produced. Check your input files and patterns.")
        sys.exit(1)

    output_df = pd.DataFrame(results, columns=['sample', 'num_gt_objects', 'num_pred_objects', 'gt_centroids_inside_pred_bboxes'])
    output_df.drop_duplicates(inplace=True)
    
    output_path = os.path.join(input_folder, 'validation_results.csv')
    try:
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except PermissionError:
        print(f"Error: Permission denied when trying to save results to {output_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
