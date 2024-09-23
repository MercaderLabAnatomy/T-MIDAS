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

def restore_images(input_folder, output_folder, model, dim_order, num_channels, restoration_model):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))
        print(f"\nImage shape: {img.shape}, dimension order: {dim_order}")
        
        # Check if image is 3D, has color channels, or is a time series
        is_3d = 'Z' in dim_order
        has_color = 'C' in dim_order
        is_time_series = 'T' in dim_order
        
        print(f"3D: {is_3d}, Multicolor: {has_color}, Time Series: {is_time_series}")

        # Determine channel axis
        channel_axis = dim_order.index('C') if has_color else None
        print(f"Color channels: {num_channels if has_color else 'Grayscale'}")

        # Prepare transpose order
        transpose_order = []
        for dim in 'ZCYX':
            if dim in dim_order:
                transpose_order.append(dim_order.index(dim))
        
        # Add time dimension at the beginning if it exists
        if is_time_series:
            transpose_order.insert(0, dim_order.index('T'))
        
        # Transpose image
        img = np.transpose(img, tuple(transpose_order))
        
        # Add channel dimension if grayscale
        if not has_color:
            img = np.expand_dims(img, axis=-1)
        
        # Process image
        if is_time_series:
            restored_img = np.zeros(img.shape, dtype=np.float32)
            for t in tqdm(range(img.shape[0]), desc="Processing time points"):
                img_t = img[t]
                if num_channels > 1:
                    img_dn = model.eval(x=[img_t for _ in range(num_channels)],
                                        channels=[[i, 0] for i in range(num_channels)],
                                        z_axis=0 if is_3d else None)
                else:
                    img_dn = model.eval(img_t, channels=[0,0], z_axis=0 if is_3d else None)
                restored_img[t] = img_dn
        else:
            if num_channels > 1:
                restored_img = model.eval(x=[img for _ in range(num_channels)],
                                          channels=[[i, 0] for i in range(num_channels)],
                                          z_axis=0 if is_3d else None)
            else:
                restored_img = model.eval(img, channels=[0,0], z_axis=0 if is_3d else None)
        
        # Normalize and stack channels
        multicolor_image = np.stack([normalize_to_uint8(np.squeeze(restored_img[i])) for i in range(len(restored_img))], axis=-1)
        print("The shape of the multicolor image is: ", multicolor_image.shape)
        
        # Reorder dimensions for ImageJ hyperstack (TZCYXS order)
        new_order = 'TZCYX'
        source_axes = [dim_order.index(d) for d in new_order if d in dim_order]
        dest_axes = list(range(len(source_axes)))
        multicolor_image = np.moveaxis(multicolor_image, source_axes, dest_axes)
        
        new_dim_order = ''.join([new_order[i] for i in dest_axes])
        print("Reordered dimensions:", new_dim_order)
        
        imwrite(os.path.join(output_folder, input_file.replace(".tif", f"_{restoration_model}.tif")),
                multicolor_image, compression='zlib', imagej=True)



# Main execution
args = parse_args()
restore_images(args.input, args.input, model, args.dim_order, args.num_channels, args.restoration_type)
