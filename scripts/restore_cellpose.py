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
import imagej


"""
Description: This script restores images using the cellpose image restoration models.

The script reads images from the input folder, 
restores them using the specified model, and saves the restored images in the same folder.

The script supports 2D and 3D images with different dimension orders.

The user can specify the restoration type (denoise, deblur, upscale) and the object type (cells, nuclei).

The cytoplasmic model usually works best.

"""


use_GPU = core.use_gpu()

def parse_args():
    parser = argparse.ArgumentParser(description="Runs automatic mask generation on images.")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    # add channels
    # parser.add_argument("--num_channels", type=int, nargs='+', default=[0,0], help="Channels to use.")
    parser.add_argument('--restoration_type',type=str, default='dn', help='Denoise, deblur or upscale? (dn/db/us)')
    parser.add_argument('--dim_order',type=str, default='ZYX', help='Dimension order of the input images.')
    parser.add_argument('--object_type',type=str, default='c', help='Cells or nuclei? (c/n)')  
    # ask for single or multichannel
    parser.add_argument('--num_channels',type=int, default='1', help='How many color channels?')
    return parser.parse_args()

args = parse_args()

# num_channels = args.num_channels
input_folder = args.input
dim_order = args.dim_order

num_channels = args.num_channels

# # check for time axis
# if 'T' in args.dim_order:
#     t_axis = args.dim_order.index('T')
# else:
#     t_axis = None

# # check for z axis
# if 'Z' in args.dim_order:
#     z_axis = args.dim_order.index('Z')
# else:
#     z_axis = None

# # check for x axis
# if 'X' in args.dim_order:
#     x_axis = args.dim_order.index('X')
# else:
#     x_axis = None

# # check for y axis
# if 'Y' in args.dim_order:
#     y_axis = args.dim_order.index('Y')
# else:
#     y_axis = None




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
elif args.restoration_type == 'us':
    if args.object_type == 'c':
        restoration_model = 'upsample_cyto3'
    elif args.object_type == 'n':
        restoration_model = 'upsample_nuclei'

else:
    print("Invalid restoration type. Choose 'dn' for denoise or 'db' for deblur.")
    exit(1) # this will stop the script, 1 means error


model = denoise.DenoiseModel(model_type=restoration_model, gpu=True)
#model = denoise.CellposeDenoiseModel(gpu=True, restore_type=restoration_model)


def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)





# def denoise_images_zyx(input_folder, output_folder, model,dim_order, num_channels):
#     input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
#     for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
#         img = imread(os.path.join(input_folder, input_file))
#         print("\n")
#         print("Check if image shape corresponds to the dim order that you have given:\n")
#         print(f"Image shape: {img.shape}, dimension order: {dim_order}")
#         print("\n")


#         if 'C' in dim_order:
#             channel_axis = dim_order.index('C')
#         else:
#             channel_axis = None  
#         # Determine if the image is 2D or 3D
#         is_3d = len(img.shape) == 3 and 'Z' in dim_order

#         if is_3d:
#             if dim_order != 'ZYX':
#                 img = np.transpose(img, (dim_order.index('Z'), dim_order.index('Y'), dim_order.index('X')))
#         else: # 2D case
#             if dim_order != 'YX':
#                 img = np.transpose(img, (dim_order.index('Y'), dim_order.index('X')))

#         # for c in tqdm(range(num_channels[0]), total=num_channels[0], desc="Processing channels"):

#         img_dn = model.eval(img, channels=num_channels, z_axis=0,channel_axis=channel_axis) 
#         # drop the last dimension
#         img_dn = np.squeeze(img_dn)
#         imwrite(os.path.join(output_folder, 
#                              input_file.replace(".tif", f"_{restoration_model}.tif")), 
#                              normalize_to_uint8(img_dn), 
#                              compression='zlib')

def denoise_images_zyx(input_folder, output_folder, model, dim_order, num_channels):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {img.shape}, dimension order: {dim_order}")
        print("\n")

        # Get the indices for each dimension
        dim_indices = {dim: dim_order.index(dim) for dim in dim_order}

        # Check if the image dimensions match the expected number of dimensions
        if len(img.shape) != len(dim_order):
            raise ValueError(f"Image shape {img.shape} does not match dimension order {dim_order}")

        # Determine the channel axis
        channel_axis = dim_indices.get('C', None)

        # Rearrange dimensions to ZYX(C) order
        if dim_order != 'ZYXC':
            transpose_order = [dim_indices['Z'], dim_indices['Y'], dim_indices['X']]
            if channel_axis is not None:
                transpose_order.append(channel_axis)
            img = np.transpose(img, tuple(transpose_order))

        if channel_axis is not None:
            img_dn = model.eval(x=[img for _ in range(num_channels)], 
                                channels=[[i, 0] for i in range(num_channels)], 
                                z_axis=0,channel_axis=channel_axis)

        else:
            img_dn = model.eval(img, channels=[0,0], z_axis=0)


        #img_dn = np.squeeze(img_dn)
        # iterate through img_dn tuple
        multicolor_image = np.stack([normalize_to_uint8(np.squeeze(img_dn[i])) for i in range(len(img_dn))], axis=-1)
        print("The shape of the multicolor image is: ", multicolor_image.shape)

        # imwrite(os.path.join(output_folder, 
        #                      input_file.replace(".tif", f"_{restoration_model}.tif")), 
        #                      multicolor_image, 
        #                      compression='zlib')
        # ImageJ hyperstack axes must be in TZCYXS order

    

        new_order = 'CZYXT' # 

        # Determine the source axes based on the current dimension order
        source_axes = [dim_order.index(d) for d in new_order if d in dim_order]

        # Determine the destination axes based on the new dimension order
        dest_axes = list(range(len(source_axes)))

        # Reorder the dimensions of the multicolor_image
        multicolor_image = np.moveaxis(multicolor_image, source_axes, dest_axes)

        # Optional: Update the new dimension order string for reference
        new_dim_order = ''.join([new_order[i] for i in dest_axes])

        print("Reordered dimensions:", new_dim_order)


        imwrite(os.path.join(output_folder, 
                             input_file.replace(".tif", f"_{restoration_model}.tif")), 
                             multicolor_image, 
                             compression='zlib', imagej=True)



def denoise_images_tzyx(input_folder, output_folder, model, dim_order, num_channels):
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
    
    for input_file in tqdm(input_files, total=len(input_files), desc="Processing images"):
        img = imread(os.path.join(input_folder, input_file))
        # print value of each dimension
        print("\n")
        print("Check if image shape corresponds to the dim order that you have given:\n")
        print(f"Image shape: {img.shape}, dimension order: {dim_order}")
        print("\n")

        if 'C' in dim_order:
            channel_axis = dim_order.index('C')
        else:
            channel_axis = None  

        # Determine if the image is 2D or 3D
        is_3d = len(img.shape) == 4 and 'Z' in dim_order

        if is_3d:
            # if dim order is not TZYX, then transpose the image
            if dim_order != 'TZYX':
                img = np.transpose(img, (dim_order.index('T'), dim_order.index('Z'), dim_order.index('Y'), dim_order.index('X')))
        else: # 2D case
            if dim_order != 'TYX':
                img = np.transpose(img, (dim_order.index('T'), dim_order.index('Y'), dim_order.index('X')))

        # with this order, t_axis should be 0
        t_axis = 0
        # Create a list to store denoised time points
        denoised_time_points = []

        for t in tqdm(range(img.shape[t_axis]), total=img.shape[t_axis], desc="Processing timepoints"):
            # Extract the current time point
            img_t = np.take(img, t, axis=t_axis)
            img_dn = model.eval(img_t, channels=num_channels, z_axis=0, channel_axis = channel_axis) # after reordering and slicing, z_axis should be 0
            # Drop the last dimension
            img_dn = np.squeeze(img_dn)

            # Stack the denoised channels for this time point
            denoised_t = np.stack(img_dn, axis=0)
            denoised_time_points.append(denoised_t)

        # Stack all denoised time points
        denoised_img = np.stack(denoised_time_points, axis=t_axis)
        imwrite(os.path.join(output_folder, 
                            input_file.replace(".tif", f"_{restoration_model}.tif")), 
                            normalize_to_uint8(denoised_img), 
                            compression='zlib')




# check for time axis and choose the right function
# if it contains T, then use denoise_images_tzyx

if 'T' in args.dim_order:
    denoise_images_tzyx(input_folder, input_folder, model, dim_order, num_channels)
else:
    denoise_images_zyx(input_folder, input_folder, model, dim_order, num_channels)