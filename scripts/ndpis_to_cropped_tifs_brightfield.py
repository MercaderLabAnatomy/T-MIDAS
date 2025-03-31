import openslide
import os
from PIL import Image
import argparse
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from skimage.filters import gaussian
from tqdm import tqdm
import pyclesperanto_prototype as cle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import tifffile as tf
import napari

"""
Description: This script reads NDPI files, extracts regions of interest (ROIs) using Mobile-SAM, and saves the ROIs as TIF files.
ROIs exceeding 4GB when saved as TIFF are skipped.

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
    parser = argparse.ArgumentParser(description='Extract ROIs from NDPI files and save them as TIF files.')
    parser.add_argument('--input', type=str, help='Path to the folder containing the NDPI(s) files.')
    parser.add_argument('--padding', type=int, default=100, help='Padding around the ROIs (default: 100).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input
output_dir = os.path.join(input_folder, "tif_files")
PADDING = args.padding
MAX_ROI_SIZE = 30000  # Maximum allowed size for ROI dimensions


# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ndpi_files = [file for file in os.listdir(input_folder) if file.endswith(".ndpi")]


def get_largest_label_id(label_image):
    label_props = regionprops(label_image)
    areas = [region.area for region in label_props]
    max_area_label = np.argmax(areas) + 1 
    return max_area_label

def get_rois(slide, output_filename):
    try:
        scaling_factor = 30
        slide_dims_downscaled = (slide.dimensions[0] / scaling_factor, slide.dimensions[1] / scaling_factor)

        thumbnail = slide.get_thumbnail(slide_dims_downscaled)
        thumbnail.save(output_filename + "_thumbnail.png")
        thumbnail_array = np.array(thumbnail)
        snapshot = thumbnail_array.copy()
        thumbnail_array = gaussian(thumbnail_array, sigma=2.0, channel_axis=-1)
        labels = np.zeros(thumbnail_array.shape[:2], dtype=np.uint32)
        print(f"Thumbnail array shape: {thumbnail_array.shape}")

        masks = mask_generator.generate(thumbnail_array) # generate masks using Mobile-SAM

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            labeled_mask = label(mask, return_num=False)
            labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max())

        largest_label_id = get_largest_label_id(labels)
        labels[labels == largest_label_id] = 0
        labels = cle.push(labels)
        dilated_labels = cle.dilate_labels(labels, None, 25.0)
        merged_dilated_labels = cle.merge_touching_labels(dilated_labels)
        merged_labels = (merged_dilated_labels * (labels > 0)).astype(np.uint32)
        labels = cle.pull(cle.connected_components_labeling_box(merged_labels))
        Image.fromarray(labels).save(output_filename + "_thumbnail_labels.png") 


        # --- Napari Viewer for interactive label editing ---
        viewer = napari.Viewer()
        viewer.add_image(snapshot, name='Thumbnail Image')
        labels_layer = viewer.add_labels(labels, name='Labels (Initial)')
        print("Napari viewer opened. Please refine the labels in the 'Labels (Initial)' layer using Napari's tools.")
        print("Once you are satisfied with the labels, close the Napari viewer window (not exit!).")
        napari.run() # blocks until viewer is closed
        labels = labels_layer.data
        print("Napari viewer closed. Continuing with processing...")

        # Upscale labels to full slide resolution
        labels_upscaled = np.zeros(slide.dimensions[::-1], dtype=np.uint32)
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                if labels[y, x] > 0:
                    start_y = int(y * scaling_factor)
                    start_x = int(x * scaling_factor)
                    labels_upscaled[start_y:start_y+scaling_factor, start_x:start_x+scaling_factor] = labels[y, x]

        props = regionprops(labels_upscaled)
        rois = []
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            minr = max(0, minr - PADDING)
            minc = max(0, minc - PADDING)
            maxr = min(slide.dimensions[1], maxr + PADDING)
            maxc = min(slide.dimensions[0], maxc + PADDING)
            rois.append((minc, minr, maxc - minc, maxr - minr))


        return rois
    except (Exception, RuntimeError) as e:
        print(f"Error processing {template_ndpi_file}: {str(e)}")
        return None


for ndpi_file in tqdm(ndpi_files, total=len(ndpi_files), desc="Processing images"):
    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
    slide = openslide.OpenSlide(os.path.join(input_folder, ndpi_file))
    rois = get_rois(slide, output_filename)
    number_of_rois = len(rois)
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        
        # Skip ROIs larger than 30000x30000
        if w > MAX_ROI_SIZE or h > MAX_ROI_SIZE:
            print(f"Skipping ROI {i+1} of {number_of_rois}: Size ({w}x{h}) exceeds maximum allowed size.")
            continue
        
        cropped_image = slide.read_region((x, y), 0, (w, h))
        cropped_image_dimensions = cropped_image.size
        cropped_image = cropped_image.convert('RGB')
        
        tif_filename = f"{output_filename}_roi_{i+1:03d}.tif"
        try:
            tf.imwrite(tif_filename, np.array(cropped_image), compression='zlib')
            print(f"ROI {i+1} of {number_of_rois} with dimensions {cropped_image_dimensions} saved as {tif_filename}")
        except Exception as e:
            print(f"Error saving ROI {i+1} of {number_of_rois}: {str(e)}")
            continue


if __name__ == "__main__":
    args = parse_args()
    # main(args)  # Uncomment and implement main() if needed
