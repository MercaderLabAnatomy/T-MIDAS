# segment blobs using cyto3 


import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import numpy as np
import os
from skimage.io import imread
from cellpose import models, core, denoise
from tifffile import imwrite
from tqdm import tqdm

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--diameter", type=float, default=40.0, help="Diameter of objects.")
    parser.add_argument("--channels", type=int, nargs='+', default=[0,0], help="Channels to use.")
    parser.add_argument('--dim_order',type=str, default='ZYX', help='Dimension order of the input images.')
    return parser.parse_args()

args = parse_args()

channels = args.channels
input_folder = args.input
diameter = args.diameter
dim_order = args.dim_order

# convert to list
if not isinstance(channels, list):
    channels = [channels]

flow_threshold = 0.4

#model = models.Cellpose(gpu=use_GPU, model_type='cyto3')
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",
             restore_type="denoise_cyto3", chan2_restore=False)

def segment_images(input_folder, output_folder, model, channels, diameter, flow_threshold):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    for input_file in tqdm(input_files, total = len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder,input_file))
        masks,flows, styles, diams = model.eval(img, diameter=diameter, 
                                                flow_threshold=flow_threshold, 
                                                channels=channels, niter=2000)
        imwrite(os.path.join(output_folder, 
                             input_file.replace(".tif", "_labels.tif")), 
                             masks.astype(np.uint32), 
                             compression='zlib')

def segment_time_series_images(input_folder, output_folder, model, channels, diameter, flow_threshold):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    for input_file in tqdm(input_files, total = len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder,input_file))
        # print value of each dimension
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {img.shape}, dimension order: {dim_order}")
        print("\n")
        # Determine if the image is 2D or 3D
        is_3d = len(image.shape) == 4 and 'Z' in dim_order
        
        if is_3d:
            if dim_order != 'TZYX':
                transpose_order = [dim_order.index(d) for d in 'TZYX']
                image = np.transpose(image, transpose_order)
        else:  # 2D case
            if dim_order != 'TYX':
                transpose_order = [dim_order.index(d) for d in 'TYX']
                image = np.transpose(image, transpose_order)

        # Pre-allocate the array for labeled time points
        labeled_time_points = np.zeros(image.shape, dtype=np.uint32)
        
        for t in tqdm(range(img.shape[0]), total=img.shape[0], desc="Processing time points"):
            # Extract the current time point
            img_t = np.take(img, t, axis=0)
            mask, flows, styles, diams = model.eval(img_t, diameter=diameter, 
                                                    flow_threshold=flow_threshold, 
                                                    channels=channels, niter=2000)
            labeled_time_points[t] = mask


        imwrite(os.path.join(output_folder, 
                             input_file.replace(".tif", "_labels.tif")), 
                             labeled_time_points.astype(np.uint32),
                             compression='zlib')

# execute segmentation
if 'T' in dim_order:
    segment_time_series_images(input_folder, input_folder, model, channels, diameter, flow_threshold)
else:
    segment_images(input_folder, input_folder, model, channels, diameter, flow_threshold)
