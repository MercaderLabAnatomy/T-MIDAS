# segment blobs using cyto3 


import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import numpy as np
import os
from skimage.io import imread
from cellpose import models, core
from tifffile import imwrite
from tqdm import tqdm

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    # add diameter
    parser.add_argument("--diameter", type=float, default=40.0, help="Diameter of objects.")
    # add channels
    parser.add_argument("--channels", type=int, nargs='+', default=[0,0], help="Channels to use.")
    return parser.parse_args()

args = parse_args()

channels = args.channels
input_folder = args.input
diameter = args.diameter

# convert to list
if not isinstance(channels, list):
    channels = [channels]

flow_threshold = 0.4

model = models.Cellpose(gpu=use_GPU, model_type='cyto3')

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



# execute segmentation
segment_images(input_folder, input_folder, model, channels, diameter, flow_threshold)

