import os
import sys
import cv2
import argparse
import pyclesperanto_prototype as cle
from tqdm import tqdm
import numpy as np
from skimage.measure import label
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import napari
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.transform import downscale_local_mean

# Add tmidas to path
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image
from tmidas.utils.argparse_utils import create_parser





"""
Description: This script runs automatic mask generation on 2D images using Mobile-SAM.
It allows users to choose between instance and semantic segmentation, interactively refine labels in Napari,
and saves the segmentation masks as uint32 TIFF files with zlib compression.
"""

model_type = "vit_t"
sam_checkpoint = "/opt/T-MIDAS/models/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)


def parse_args():
    parser = create_parser("Runs automatic mask generation on images with interactive label refinement and segmentation type choice.")
    return parser.parse_args()

args = parse_args()

image_folder = args.input
SIZE_LIMIT = 1024 * 1024 * 3 



def calculate_downscale_factor(num_pixels, target_pixels=SIZE_LIMIT):
    """Calculate a downscale factor that shrinks the image to target_pixels or less."""
    return np.sqrt(target_pixels / num_pixels)

# def calculate_downscale_factor(num_pixels, target_pixels=SIZE_LIMIT):
    # return int(np.ceil(np.sqrt(num_pixels / target_pixels)))

def get_largest_label_id(label_image):
    label_props = regionprops(label_image)
    areas = [region.area for region in label_props]
    max_area_label = np.argmax(areas) + 1 
    return max_area_label

# def fast_upscale_uint32(image, new_shape):
#     zoom_factors = np.array(new_shape) / np.array(image.shape)
#     return image[np.rint(np.arange(new_shape[0])/zoom_factors[0]).astype(int)][:, np.rint(np.arange(new_shape[1])/zoom_factors[1]).astype(int)]



def fast_upscale_uint32_cv2(image, new_shape):
    # Split the uint32 image into four uint8 channels
    channels = [(image >> (8*i)) & 0xFF for i in range(4)]
    
    # Resize each channel
    resized_channels = [cv2.resize(channel.astype(np.uint8), (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST) for channel in channels]
    
    # Combine the channels back into a uint32 image
    result = sum(resized_channels[i].astype(np.uint32) << (8*i) for i in range(4))
    
    return result



def process_image(image_path):
    try:
        image_original = read_image(image_path)
        # make sure that the image is 8bit
        if image_original.dtype != np.uint8:
            image_original = (image_original / np.amax(image_original) * 255).astype(np.uint8)


        # mask gen expects third dim color channel
        if len(image_original.shape) == 2:
            height, width = image_original.shape
            size = height * width 
            if size > SIZE_LIMIT:
                downscale_factor = calculate_downscale_factor(size)
                # image = downscale_local_mean(image_original, (factor, factor)).astype(np.uint8)
                image = cv2.resize(image_original, (int(width * downscale_factor), int(height * downscale_factor)), interpolation=cv2.INTER_AREA)
                #image = resize(image_original, (int(height * downscale_factor), int(width * downscale_factor)), anti_aliasing=True)
                downscaled = True
                print(f"Downscaled image to {image.shape}")
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            else:
                downscaled = False
                image = image_original[:, :, np.newaxis].repeat(3, axis=2)
        else:
            # Handle color images (3D)
            height, width = image_original.shape[:2]
            size = height * width
            if size > SIZE_LIMIT:
                downscale_factor = calculate_downscale_factor(size)
                image = cv2.resize(image_original, (int(width * downscale_factor), int(height * downscale_factor)), interpolation=cv2.INTER_AREA)
                downscaled = True
                print(f"Downscaled image to {image.shape}")
            else:
                downscaled = False
                image = image_original

        # image_pre = cle.push(image)
        # #image_pre = cle.gaussian_blur(image_pre, None, 1.0, 1.0, 0.0)
        # # # image = cle.gamma_correction(image, None, 1.0)
        # # # #image = cle.top_hat_box(image, None, 10.0, 10.0, 0)
        # image_pre = cle.detect_label_edges(image_pre)
        # image_pre = cle.pull(image_pre)

        masks = mask_generator.generate(image)  # generate masks using Mobile-SAM

        labels = np.zeros(image.shape[:2], dtype=np.uint32) # Use uint32 for labels

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            labeled_mask = label(mask, return_num=False)
            labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max())
        largest_label_id = get_largest_label_id(labels)
        #labels[labels == largest_label_id] = 0

        # --- Napari Viewer for interactive label editing ---
        viewer = napari.Viewer()
        viewer.add_image(image.astype(np.uint8), name='Image')
        labels_layer = viewer.add_labels(labels, name=f'Labels')
        print(f"Napari viewer opened for segmentation...") # Debug print
        print("Once you are satisfied with the labels, close the Napari viewer window.")
        print("About to call napari.run()") # Debug print IMMEDIATELY BEFORE napari.run()
        napari.run() # blocks until viewer is closed
        user_modified_labels = labels_layer.data
        print(f"Napari viewer closed. Continuing with processing...")


        if downscaled:
            user_modified_labels = fast_upscale_uint32_cv2(user_modified_labels, (height, width))
            # user_modified_labels = resize(user_modified_labels, (height, width), order=0, anti_aliasing=False).astype(np.uint32)
        print(f"Upscaled label image to {user_modified_labels.shape}")

        return user_modified_labels

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def save_image(image, filename):
    image_uint32 = image.astype(np.uint32)
    write_image(image_uint32, filename)


for filename in tqdm(os.listdir(image_folder), total = len(os.listdir(image_folder)), desc="Processing images"):
    if not filename.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        continue

    labeled_image = process_image(os.path.join(image_folder, filename))

    if labeled_image is not None:
        output_file = os.path.join(image_folder, filename.replace(".tif", "_labels.tif"))
        save_image(labeled_image, output_file)
        del labeled_image
        torch.cuda.empty_cache()
        print(f"Saved {output_file}")
