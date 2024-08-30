import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
from skimage import img_as_uint
from cellpose import models, core, denoise
from tqdm import tqdm

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument('--restoration_type', type=str, default='dn', help='Denoise, deblur or upscale? (dn/db/us)')
    parser.add_argument('--dim_order', type=str, default='ZYX', help='Dimension order of the input images.')
    parser.add_argument('--object_type', type=str, default='c', help='Cells or nuclei? (c/n)')
    return parser.parse_args()

args = parse_args()

input_folder = args.input
dim_order = args.dim_order

# Choose model based on restoration type and object type
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
elif args.restoration_type == 'us':
    if args.object_type == 'c':
        restoration_model = 'upsample_cyto3'
    elif args.object_type == 'n':
        restoration_model = 'upsample_nuclei'
else:
    print("Invalid restoration type. Choose 'dn' for denoise, 'db' for deblur, or 'us' for upsample.")
    exit(1)

model = denoise.DenoiseModel(model_type=restoration_model, gpu=True)

def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def denoise_images(input_folder, output_folder, model, dim_order):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        try:
            img = imread(os.path.join(input_folder, input_file))
            print("\n")
            print("Check if image shape corresponds to the dim order that you have given:\n")
            print(f"Image shape: {img.shape}, dimension order: {dim_order}")
            print("\n")

            is_3d = len(img.shape) == 3 and 'Z' in dim_order

            if is_3d:
                if dim_order != 'ZYX':
                    img = np.transpose(img, (dim_order.index('Z'), dim_order.index('Y'), dim_order.index('X')))
            else:  # 2D case
                if dim_order != 'YX':
                    img = np.transpose(img, (dim_order.index('Y'), dim_order.index('X')))

            img_dn = model.eval(img, channels=[0, 0], z_axis=0)
            img_dn = np.squeeze(img_dn)
            imwrite(os.path.join(output_folder, 
                                 input_file.replace(".tif", f"_{restoration_model}.tif")), 
                                 normalize_to_uint8(img_dn), 
                                 compression='zlib')
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

if 'T' in args.dim_order:
    denoise_images_tzyx(input_folder, input_folder, model, dim_order)
else:
    denoise_images(input_folder, input_folder, model, dim_order)
