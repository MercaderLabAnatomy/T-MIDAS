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
    parser.add_argument('--cropping_template_channel_name', type=str, help='Enter the channel name that represents the cropping template (hearts = FITC).')
    parser.add_argument('--padding', type=int, default=10, help='Padding around the ROIs (default: 10).')
    # parser.add_argument('--level', type=int, help='Enter the resolution level of the NDPI image (0 = highest resolution, 1 = second highest resolution).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input

PADDING = args.padding

# ask for the channel that contains the cropping template
CROPPING_TEMPLATE_CHANNEL_NAME = args.cropping_template_channel_name


output_dir = input_folder + "/tif_files"

# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ndpis_files = []
for file in os.listdir(input_folder):
    if file.endswith(".ndpis"):
        ndpis_files.append(file)


def get_ndpi_filenames(ndpis_file):
    ndpi_files = []
    with open(ndpis_file, 'r') as f:
        for line in f:
            if line.endswith('.ndpi\n'):
                # extract substring after "="
                line = line.split("=")[1]
                # save to list            
                ndpi_files.append(line.rstrip('\n'))
        # close file
        f.close()
    return ndpi_files

def get_largest_label_id(label_image):
    label_props = regionprops(label_image)
    areas = [region.area for region in label_props]
    max_area_label = np.argmax(areas) + 1 
    return max_area_label


def get_rois(template_ndpi_file,output_filename):

    slide = openslide.OpenSlide(os.path.join(input_folder, template_ndpi_file))
    scaling_factor = 100
    slide_dims_downscaled = (slide.dimensions[0] / scaling_factor, slide.dimensions[1] / scaling_factor)
    thumbnail = slide.get_thumbnail(slide_dims_downscaled)
    # thumbnail = thumbnail.convert('L')
    # labeled_thumbnail = nsbatwm.gauss_otsu_labeling(thumbnail, 10.0)
    thumbnail.save(output_filename + "_thumbnail.png")
    thumbnail_array = np.array(thumbnail)
    thumbnail_shape = thumbnail_array.shape[:2]
    labels = np.zeros(thumbnail_shape, dtype=np.uint32)
    masks = mask_generator.generate(thumbnail_array) # generate masks using Mobile-SAM
    for i, mask_data in enumerate(masks):
          mask = mask_data["segmentation"]
          labeled_mask = label(mask, return_num=False)
          labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max()) 

    # get the id of the largest label and delete that label
    largest_label_id = get_largest_label_id(labels)
    # set that id to zero 
    labels[labels == largest_label_id] = 0

    # dilate labels
    labels = cle.push(labels)
    labels = cle.dilate_labels(labels, None, 10.0)
    labels = cle.merge_touching_labels(labels)
    labels = cle.pull(labels)

    Image.fromarray(labels).save(output_filename + "_thumbnail_labels.png")
    props = regionprops(labels)
    rois = []
    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - PADDING)
        minc = max(0, minc - PADDING)
        maxr = min(thumbnail.height, maxr + PADDING)
        maxc = min(thumbnail.width, maxc + PADDING)
        rois.append((minc*scaling_factor, minr*scaling_factor, (maxc-minc)*scaling_factor, (maxr-minr)*scaling_factor))
          
    return rois


for ndpis_file in ndpis_files:


    ndpi_files = get_ndpi_filenames(os.path.join(input_folder, ndpis_file))
    CROPPING_TEMPLATE_CHANNEL = [ndpi_file for ndpi_file in ndpi_files if CROPPING_TEMPLATE_CHANNEL_NAME in ndpi_file][0]
    output_filename_thumbnail = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpis_file))[0])
    rois = get_rois(CROPPING_TEMPLATE_CHANNEL,output_filename_thumbnail)
    number_of_rois = len(rois)

    for ndpi_file in tqdm(ndpi_files, total = len(ndpi_files), desc="Processing images"):
        if ndpi_file.endswith(".ndpi"):

            output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
            slide = openslide.OpenSlide(os.path.join(input_folder, ndpi_file))
            for i, roi in enumerate(rois):
                x, y, w, h = roi

                cropped_image = slide.read_region((x, y), 0, (w, h))
                cropped_image_dimensions = cropped_image.size
                print("ROI %d of %d with dimensions %s saved as %s" % (i+1, number_of_rois, cropped_image_dimensions, output_filename + "_roi_0" + str(i+1) + ".tif"))
                cropped_image = cropped_image.convert('L')
                #cropped_image.save(output_filename + "_roi_0" + str(i+1) + ".tif")
                tf.imwrite(output_filename + "_roi_0" + str(i+1) + ".tif", cropped_image, compression='zlib')

