import openslide
import os
from PIL import Image, ImageOps
import argparse
from skimage.measure import regionprops
# from cucim.skimage.measure import regionprops as regionprops_gpu
from tqdm import tqdm
import pyclesperanto_prototype as cle  # version 0.24.2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from skimage.measure import label
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

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
    parser.add_argument('--padding', type=int, default=10, help='Padding around the ROIs (default: 10).')
    return parser.parse_args()

args = parse_args()

input_folder = args.input

output_dir = input_folder + "/tif_files"


PADDING = args.padding


# make output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


ndpi_files = []
for file in os.listdir(input_folder):
    if file.endswith(".ndpi"):
        ndpi_files.append(file)


# def save_image(image, filename):
#     image_uint32 = image.astype(np.uint32)
#     imwrite(filename, image_uint32, compression='zlib')

def get_rois(slide, output_filename):

    scaling_factor = 100
    slide_dims_downscaled = (slide.dimensions[0] / scaling_factor, slide.dimensions[1] / scaling_factor)
    thumbnail = slide.get_thumbnail(slide_dims_downscaled)
    thumbnail.save(output_filename + "_thumbnail.png")
    thumbnail_array = np.array(thumbnail)
    thumbnail_shape = thumbnail_array.shape[:2]
    labels = np.zeros(thumbnail_shape, dtype=np.uint32)
    masks = mask_generator.generate(thumbnail_array) # generate masks using Mobile-SAM
    for i, mask_data in enumerate(masks):
          mask = mask_data["segmentation"]
          labeled_mask = label(mask, return_num=False)
          labels[labeled_mask > 0] = labeled_mask[labeled_mask > 0] + (i * labeled_mask.max())
    props = regionprops(labels)
    areas = [region.area for region in props]
    max_area_label = np.argmax(areas) + 1 
    labels[labels == max_area_label] = 0     # remove the largest label
    labels = cle.dilate_labels(labels, None, 2.0)
    labels = cle.merge_touching_labels(labels)
    labels = cle.pull(labels)
    Image.fromarray(labels).save(output_filename + "_thumbnail_labels.png")
    # save_image(labels, output_filename + "_labels.tif")

    rois = []
    for i, prop in enumerate(props):
        minr, minc, maxr, maxc = prop.bbox
        minr = max(0, minr - PADDING) # 
        minc = max(0, minc - PADDING)
        maxr = min(thumbnail.height, maxr + PADDING)
        maxc = min(thumbnail.width, maxc + PADDING)
        rois.append((minc*scaling_factor, minr*scaling_factor, (maxc-minc)*scaling_factor, (maxr-minr)*scaling_factor))
    
    # drop rois that are 5x the size of the median roi
    median_roi_size = sorted([roi[2]*roi[3] for roi in rois])[int(len(rois)/2)]
    rois = [roi for roi in rois if roi[2]*roi[3] < 5*median_roi_size]    



    # Calculate the centroids of the ROIs
    centroids = [(x + w // 2, y + h // 2) for x, y, w, h in rois]

    # Calculate the distances between all pairs of centroids
    distances = []
    for i in range(len(rois)):
        for j in range(i+1, len(rois)):
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            distances.append(distance)

    # Calculate the median distance
    median_distance = np.median(distances)

    # Merge ROIs that are closer than a third of the median distance
    for i in range(len(rois)):
        for j in range(i+1, len(rois)):
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if distance < median_distance / 5:
                # Merge the ROIs
                x = min(rois[i][0], rois[j][0])
                y = min(rois[i][1], rois[j][1])
                w = max(rois[i][0] + rois[i][2], rois[j][0] + rois[j][2]) - x
                h = max(rois[i][1] + rois[i][3], rois[j][1] + rois[j][3]) - y
                rois[i] = (x, y, w, h)
                rois[j] = (0, 0, 0, 0)



    rois = [roi for roi in rois if roi[2] > 0 and roi[3] > 0]
    # len(rois)
      
    return rois


for ndpi_file in tqdm(ndpi_files, total = len(ndpi_files), desc="Processing images"):

    output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0])
    slide = openslide.OpenSlide(os.path.join(input_folder, ndpi_file))
    rois = get_rois(slide,output_filename)
    number_of_rois = len(rois)
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        cropped_image = slide.read_region((x, y), 0, (w, h))
        cropped_image_dimensions = cropped_image.size
        print("ROI %d of %d with dimensions %s saved as %s" % (i+1, number_of_rois, 
                                                               cropped_image_dimensions, 
                                                               output_filename + "_roi_0" + str(i+1) + ".tif"))
        cropped_image = cropped_image.convert('RGB')
        cropped_image.save(output_filename + "_roi_0" + str(i+1) + ".tif", compression="tiff_deflate")

