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
    # add channels
    parser.add_argument("--num_channels", type=int, nargs='+', default=[0,0], help="Channels to use.")
    parser.add_argument('--restoration_type',type=str, default='dn', help='Denoise or deblur? (dn/db)')
    parser.add_argument('--object_type',type=str, default='c', help='Cells or nuclei? (c/n)')
    return parser.parse_args()

args = parse_args()

num_channels = args.num_channels
input_folder = args.input


# choose model based on restoration and type
if args.restoration_type == 'dn':
    if args.object_type == 'c':
        restoration_model = 'denoise_cyto3'
    elif args.object_type == 'n':
        restoration_model = 'denoise_nuclei'
elif args.restoration_type == 'db':
    if args.object_type == 'c':
        restoration_model = 'deblur_cyto3'
    elif args.object_type == 'n':
        restoration_model = 'deblur_nuclei'
else:
    print("Invalid restoration type. Choose 'dn' for denoise or 'db' for deblur.")
    exit(1) # this will stop the script, 1 means error


model = denoise.DenoiseModel(model_type=restoration_model, gpu=True)


def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)





def denoise_images(input_folder, output_folder, model, num_channels):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))

        for c in tqdm(range(num_channels[0]), total=num_channels[0], desc="Processing channels"):

            img_dn = model.eval(img, channels=[c, 0], z_axis=0)#,channel_axis=4) # z_axis should be user input
            # drop the last dimension
            img_dn = np.squeeze(img_dn)
            imwrite(os.path.join(output_folder, 
                                 input_file.replace(".tif", f"_{restoration_model}.tif")), 
                                 normalize_to_uint8(img_dn), 
                                 compression='zlib')



# execute segmentation
denoise_images(input_folder, input_folder, model, num_channels)

# CLI usage: python scripts/denoise_cellpose.py --input data --channels 3