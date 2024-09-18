import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
from cellpose import models, core, denoise
from tqdm import tqdm

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument('--restoration_type', type=str, default='dn', help='Denoise, deblur or upscale? (dn/db/us)')
    parser.add_argument('--dim_order', type=str, default='ZYX', help='Dimension order of the input images.')
    parser.add_argument('--object_type', type=str, default='c', help='Cells or nuclei? (c/n)')
    parser.add_argument('--num_channels', type=int, default=1, help='How many color channels?')
    return parser.parse_args()

args = parse_args()
input_folder = args.input
dim_order = args.dim_order
num_channels = args.num_channels

if args.restoration_type == 'dn':
    restoration_model = 'denoise_cyto3' if args.object_type == 'c' else 'denoise_nuclei'
elif args.restoration_type == 'db':
    restoration_model = 'deblur_cyto3' if args.object_type == 'c' else 'deblur_nuclei'
elif args.restoration_type == 'us':
    restoration_model = 'upsample_cyto3' if args.object_type == 'c' else 'upsample_nuclei'
else:
    print("Invalid restoration type. Choose 'dn' for denoise, 'db' for deblur, or 'us' for upsample.")
    exit(1)

model = denoise.DenoiseModel(model_type=restoration_model, gpu=True)

def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def denoise_images_zyx(input_folder, output_folder, model, dim_order, num_channels):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))
        print(f"\nImage shape: {img.shape}, dimension order: {dim_order}")
        
        dim_indices = {dim: i for i, dim in enumerate(dim_order)}
        
        is_2d = 'Z' not in dim_indices
        channel_axis = dim_indices.get('C', None)
        
        if is_2d:
            transpose_order = [dim_indices['Y'], dim_indices['X']]
            if channel_axis is not None:
                transpose_order.append(channel_axis)
            img = np.transpose(img, tuple(transpose_order))
            img = np.expand_dims(img, axis=0)  # Add Z dimension
        else:
            transpose_order = [dim_indices['Z'], dim_indices['Y'], dim_indices['X']]
            if channel_axis is not None:
                transpose_order.append(channel_axis)
            img = np.transpose(img, tuple(transpose_order))
        
        if channel_axis is not None:
            img_dn = model.eval(x=[img for _ in range(num_channels)],
                                channels=[[i, 0] for i in range(num_channels)],
                                z_axis=0, channel_axis=-1)
        else:
            img_dn = model.eval(img, channels=[0,0], z_axis=0)
        
        multicolor_image = np.stack([normalize_to_uint8(np.squeeze(img_dn[i])) for i in range(len(img_dn))], axis=-1)
        print("The shape of the multicolor image is: ", multicolor_image.shape)
        
        # Reorder dimensions for ImageJ hyperstack (TZCYXS order)
        new_order = 'CZYXT'
        source_axes = [dim_order.index(d) for d in new_order if d in dim_order]
        dest_axes = list(range(len(source_axes)))
        multicolor_image = np.moveaxis(multicolor_image, source_axes, dest_axes)
        
        new_dim_order = ''.join([new_order[i] for i in dest_axes])
        print("Reordered dimensions:", new_dim_order)
        
        imwrite(os.path.join(output_folder, input_file.replace(".tif", f"_{restoration_model}.tif")),
                multicolor_image, compression='zlib', imagej=True)

# Main execution
args = parse_args()
denoise_images_zyx(args.input, args.input, model, args.dim_order, args.num_channels)
