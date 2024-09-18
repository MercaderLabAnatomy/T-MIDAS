import openslide
import os
import numpy as np
import argparse
import tifffile as tf
from skimage.measure import regionprops
from tqdm import tqdm
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import torch
from skimage.measure import label
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import pyclesperanto_prototype as cle

"""
Description: This script reads NDPI files, extracts regions of interest (ROIs) using Mobile-SAM, 
and saves the ROIs as grayscale TIF files based on the color channel order in the NDPIS file.

The script uses the openslide library to read the NDPI files and the Mobile-SAM model to extract the ROIs.

The output TIF files are saved in a folder named "tif_files" in the same directory as the input NDPI files.
"""

model_type = "vit_t"
sam_checkpoint = "/opt/T-MIDAS/models/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

def parse_args():
    parser = argparse.ArgumentParser(description='Extract ROIs from NDPI files and save them as grayscale TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the NDPI(s) files.')
    parser.add_argument('--cropping_template_channel_name', type=str, help='Enter the channel name that represents the cropping template (hearts = FITC).')
    parser.add_argument('--padding', type=int, default=10, help='Padding around the ROIs (default: 10).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input
PADDING = args.padding
CROPPING_TEMPLATE_CHANNEL_NAME = args.cropping_template_channel_name

output_dir = os.path.join(input_folder, "tif_files")

# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ndpis_files = [file for file in os.listdir(input_folder) if file.endswith(".ndpis")]

def get_ndpi_filenames(ndpis_file):
    ndpi_files = []
    with open(ndpis_file, 'r') as f:
        for i, line in enumerate(f):
            if line.endswith('.ndpi\n'):
                line = line.split("=")[1]
                ndpi_files.append((line.rstrip('\n'), i))
    return ndpi_files

def get_largest_label_id(label_image):
    label_props = regionprops(label_image)
    areas = [region.area for region in label_props]
    max_area_label = np.argmax(areas) + 1 
    return max_area_label

def get_rois(template_ndpi_file, output_filename):
    slide = openslide.OpenSlide(os.path.join(input_folder, template_ndpi_file))
    scaling_factor = 50
    slide_dims_downscaled = (slide.dimensions[0] / scaling_factor, slide.dimensions[1] / scaling_factor)
    thumbnail = slide.get_thumbnail(slide_dims_downscaled)
    thumbnail.save(output_filename + "_thumbnail.png")
    thumbnail_array = np.array(thumbnail)
    thumbnail_array = cle.push(thumbnail_array)
    thumbnail_array = cle.gaussian_blur(thumbnail_array, None, 1.0, 1.0, 0.0)
    thumbnail_array = cle.top_hat_box(thumbnail_array, None, 10.0, 10.0, 0)
    thumbnail_array = cle.pull(thumbnail_array)
    thumbnail_shape = thumbnail_array.shape[:2]
    labels = np.zeros(thumbnail_shape, dtype=np.uint32)
    masks = mask_generator.generate(thumbnail_array)
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        labeled_mask = label(mask, return_num=False)
        labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max()) 

    largest_label_id = get_largest_label_id(labels)
    labels[labels == largest_label_id] = 0

    labels = cle.push(labels)
    labels = cle.dilate_labels(labels, None, 10.0)
    labels = cle.merge_touching_labels(labels)
    labels = cle.pull(labels)

    Image.fromarray(labels).save(output_filename + "_thumbnail_labels.png")
    props = regionprops(labels)
    rois = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - PADDING)
        minc = max(0, minc - PADDING)
        maxr = min(thumbnail.height, maxr + PADDING)
        maxc = min(thumbnail.width, maxc + PADDING)
        rois.append((minc*scaling_factor, minr*scaling_factor, (maxc-minc)*scaling_factor, (maxr-minr)*scaling_factor))
          
    return rois

def normalize_to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)

for ndpis_file in ndpis_files:
    ndpi_files = get_ndpi_filenames(os.path.join(input_folder, ndpis_file))
    CROPPING_TEMPLATE_CHANNEL = [ndpi_file for ndpi_file, _ in ndpi_files if CROPPING_TEMPLATE_CHANNEL_NAME in ndpi_file][0]
    output_filename_thumbnail = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpis_file))[0])
    rois = get_rois(CROPPING_TEMPLATE_CHANNEL, output_filename_thumbnail)
    number_of_rois = len(rois)

    for ndpi_file, channel_index in tqdm(ndpi_files, total=len(ndpi_files), desc="Processing images"):
        if ndpi_file.endswith(".ndpi"):
            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
            slide = openslide.OpenSlide(os.path.join(input_folder, ndpi_file))
            for i, roi in enumerate(rois):
                x, y, w, h = roi
                cropped_image = slide.read_region((x, y), 0, (w, h))
                cropped_image_array = np.array(cropped_image)

                # Extract the appropriate channel based on the file order
                if channel_index < 3:  # R, G, or B
                    grayscale_image = cropped_image_array[:,:,channel_index]
                else:  # other  channels
                    grayscale_image = cropped_image_array[:,:,0]

                grayscale_image = normalize_to_uint8(grayscale_image)
                
                print(f"ROI {i+1} of {number_of_rois} with dimensions {grayscale_image.shape} saved as {output_filename}_roi_0{i+1}.tif")
                tf.imwrite(f"{output_filename}_roi_0{i+1}.tif", grayscale_image, compression='zlib')

if __name__ == "__main__":
    args = parse_args()
    # Your main execution code here
