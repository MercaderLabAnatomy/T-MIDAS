import os
import argparse
import numpy as np
from skimage.io import imread
from tifffile import imwrite
from cellpose import models, core, denoise
from tqdm import tqdm
import gc

use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument('--restoration_type', type=str, choices=['dn', 'db', 'us', 'all'], 
                        required=True, help='Denoise (dn), deblur (db), upscale (us), or all models?')
    parser.add_argument('--dim_order', type=str, default='ZYX', help='Dimension order of the input images.')
    parser.add_argument('--object_type', type=str, default='c', help='Cells or nuclei? (c/n)')
    parser.add_argument('--num_channels', type=int, default=1, help='How many color channels?')
    parser.add_argument('--diameter', type=float, required=True, help='Diameter for the DenoiseModel.')
    return parser.parse_args()



args = parse_args()
input_folder = args.input
dim_order = args.dim_order
num_channels = args.num_channels
diameter = args.diameter

if args.restoration_type == 'all':
    restoration_types = ['dn', 'db', 'us']
else:
    restoration_types = [args.restoration_type]

models_dict = {
    'dn': 'denoise_cyto3' if args.object_type == 'c' else 'denoise_nuclei',
    'db': 'deblur_cyto3' if args.object_type == 'c' else 'deblur_nuclei',
    'us': 'upsample_cyto3' if args.object_type == 'c' else 'upsample_nuclei'
}


def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

def restore_images(input_folder, output_folder, restoration_types, dim_order, num_channels, diameter):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))
        print(f"\nImage shape: {img.shape}, dimension order: {dim_order}")
        
        # Check if image is 3D, has color channels, or is a time series
        is_3d = 'Z' in dim_order
        has_color = 'C' in dim_order
        is_time_series = 'T' in dim_order
        
        print(f"3D: {is_3d}, Multicolor: {has_color}, Time Series: {is_time_series}")

        restored_img = img  # Start with original image

        for restoration_type in restoration_types:
            restoration_model = models_dict[restoration_type]
            model = denoise.DenoiseModel(model_type=restoration_model, gpu=True)

            # Process image
            if is_time_series:
                #temp_restored_img = np.zeros_like(restored_img, dtype=np.float32)
                for t in tqdm(range(restored_img.shape[0]), desc=f"Running {models_dict[restoration_type]}"):
                    img_t = restored_img[t]

                    if num_channels > 1:
                        img_dn = model.eval(x=[img_t for _ in range(num_channels)],
                                            channels=[[i, 0] for i in range(num_channels)],
                                            z_axis=0 if is_3d else None,
                                            diameter=diameter)
                        if t == 0:
                            spatial_shape = np.squeeze(img_dn).shape
                            temp_restored_img = np.zeros((restored_img.shape[dim_order.index('T')],) + spatial_shape,
                                                        dtype=np.float32)
                            


                    else:
                        img_dn = model.eval(img_t, channels=[0,0], z_axis=0 if is_3d else None,
                                            diameter=diameter)
                        

                        # Initialize temp_restored_img on the first iteration
                        if t == 0:
                            spatial_shape = np.squeeze(img_dn).shape  # Get spatial dimensions from img_dn
                            temp_restored_img = np.zeros((restored_img.shape[dim_order.index('T')],) + spatial_shape,
                                                           dtype=np.float32)



                    temp_restored_img[t] = np.squeeze(img_dn)  # Squeeze any extra dimensions
                restored_img = temp_restored_img
            else:
                if num_channels > 1:
                    restored_img = model.eval(x=[restored_img for _ in range(num_channels)],
                                              channels=[[i, 0] for i in range(num_channels)],
                                              z_axis=0 if is_3d else None,
                                              diameter=diameter)
                else:
                    restored_img = model.eval(restored_img, channels=[0,0], z_axis=0 if is_3d else None,
                                               diameter=diameter)
                restored_img = np.squeeze(restored_img)  # Squeeze any extra dimensions
            

        processed_image = normalize_to_uint8(restored_img)
        
        print("The shape of the restored image is: ", processed_image.shape)
        if args.restoration_type == 'all':
            imwrite(os.path.join(output_folder, input_file.replace(".tif", "_restored.tif")),
                    processed_image, compression='zlib', imagej=True)
        else:
            # Save the processed image with a suffix indicating the restoration type
            imwrite(os.path.join(output_folder, input_file.replace(".tif", f"_{restoration_model}.tif")),
                    processed_image, compression='zlib', imagej=True)
        gc.collect()


# Main execution
restore_images(args.input, args.input, restoration_types, dim_order, num_channels, diameter)
