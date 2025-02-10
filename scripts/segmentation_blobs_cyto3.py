import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
from cellpose import models, core
from tqdm import tqdm

"""
Description: This script runs automatic instance segmentation on images. 

The script reads images from the input folder, processes them using Cellpose 3, and saves the masks in the same folder.


"""

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--diameter", type=float, default=40.0, help="Diameter of objects.")
    parser.add_argument("--channels", type=int, nargs='+', default=[0,0], help="Channels to use.")
    parser.add_argument('--dim_order', type=str, default='ZYX', help='Dimension order of the input images.')
    parser.add_argument("--model_type", type=str, default='cyto3',
                    choices=['cyto', 'cyto2', 'cyto3', 'nuclei'],
                    help="Model type: 'cyto'/'cyto2'/'cyto3' for cells, 'nuclei' for nuclei")
    return parser.parse_args()

def process_image(input_file, input_folder, model, channels, diameter, flow_threshold, dim_order):
    img = imread(os.path.join(input_folder, input_file))
    
    print("\nCheck if image shape corresponds to the dim order that you have given:")
    print(f"Image shape: {img.shape}, dimension order: {dim_order}\n")
    
    is_3d = len(img.shape) == (4 if 'T' in dim_order else 3) and 'Z' in dim_order
    
    if 'T' in dim_order:
        transpose_order = [dim_order.index(d) for d in 'TZYX' if d in dim_order]
    else:
        transpose_order = [dim_order.index(d) for d in 'ZYX' if d in dim_order]
    
    img = np.transpose(img, transpose_order)
    
    if 'T' in dim_order:
        labeled_time_points = np.zeros(img.shape, dtype=np.uint32)
        for t in tqdm(range(img.shape[0]), desc="Processing time points"):
            img_t = img[t]
            mask, _, _, _ = model.eval(img_t, diameter=diameter, flow_threshold=flow_threshold, 
                                       channels=channels, niter=2000, z_axis=0 if is_3d else None, do_3D=is_3d)
            labeled_time_points[t] = mask
        result = labeled_time_points
    else:
        result, _, _, _ = model.eval(img, diameter=diameter, flow_threshold=flow_threshold, 
                                     channels=channels, niter=2000, z_axis=0 if is_3d else None, do_3D=is_3d)
    
    output_file = os.path.join(input_folder, input_file.replace(".tif", "_labels.tif"))
    imwrite(output_file, result.astype(np.uint32), compression='zlib')

def check_image_size(input_file, input_folder, max_pixels=4000000):
    """Check if a 2D image is larger than max_pixels squared."""
    img = imread(os.path.join(input_folder, input_file))
    if len(img.shape) == 3 and 'Z' not in args.dim_order and 'T' not in args.dim_order:
        # For 2D images
        total_pixels = img.shape[0] * img.shape[1]
        return total_pixels > max_pixels
    else:
        return False

def main():
    global args
    args = parse_args()
    
    channels = args.channels if isinstance(args.channels, list) else [args.channels]
    input_folder = args.input
    diameter = args.diameter
    dim_order = args.dim_order
    
    flow_threshold = 0.4
    model = models.Cellpose(gpu=use_GPU, model_type=args.model_type)
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, desc="Processing images"):
        if check_image_size(input_file, input_folder):
            print(f"Skipping {input_file} as it exceeds the maximum size of 4,000,000 pixels squared.")
            continue
        process_image(input_file, input_folder, model, channels, diameter, flow_threshold, dim_order)

if __name__ == "__main__":
    main()
