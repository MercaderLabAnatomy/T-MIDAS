# denoise images using cellpose


import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
import numpy as np
import os
from skimage.io import imread
from skimage import img_as_uint
from cellpose import models, core, denoise
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

#model = models.Cellpose(gpu=use_GPU, model_type='cyto3')
model = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True)


def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)





def denoise_images(input_folder, output_folder, model, channels, diameter):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    for input_file in tqdm(input_files, total = len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder,input_file))
        img_dn= model.eval(img, diameter=diameter, 
                                                channels=channels,z_axis=0)
        # drop the last dimension
        img_dn = np.squeeze(img_dn)
        imwrite(os.path.join(output_folder, 
                             input_file.replace(".tif", "_denoised.tif")), 
                             normalize_to_uint8(img_dn), 
                             compression='zlib')



# execute segmentation
denoise_images(input_folder, input_folder, model, channels, diameter)

# CLI usage: python scripts/denoise_cellpose.py --input data --diameter 40 --channels 0 0