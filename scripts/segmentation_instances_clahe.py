import os
import tifffile as tf
import argparse
from skimage import exposure, util
import cupy as cp
from skimage.io import imshow
from cucim.skimage.exposure import equalize_adapthist
from cucim.skimage import morphology, measure
from cucim.skimage.filters import gaussian
from cucim.skimage.filters.thresholding import threshold_otsu
from cucim.skimage.segmentation import relabel_sequential
import numpy as np
import matplotlib.pyplot as plt
from cucim.skimage.color import label2rgb
from cucim.skimage.measure import label

def tuple_of_floats(arg):
    return tuple(map(float, arg.split(',')))


# Argument Parsing
parser = argparse.ArgumentParser(description="Segments CLAHE images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--kernel_size", type=int, required=True, help="Defines the shape of contextual regions.")
parser.add_argument("--clip_limit", type=float, required=True, help="Defines the contrast limit for localised histogram equalisation.")
parser.add_argument("--nbins", type=int, required=True, help="Number of bins for the histogram.")
parser.add_argument("--outline_sigma", type=float, default=1.0, help="Defines the sigma for the gauss-otsu-labeling.")
args = parser.parse_args()

input_folder = args.input




mask_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('_labels.tif')]
intensity_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
mask_files.sort()
intensity_files.sort()


# mask = "/media/geffjoldblum/DATA/tests/instance_segmentation_clahe/FITC/C2-Ki67_well_01 - 2022-11-25 11_tile_01_labels.tif"
# image = "/media/geffjoldblum/DATA/tests/instance_segmentation_clahe/FITC/C2-Ki67_well_01 - 2022-11-25 11_tile_01.tif"
# kernel_size = 32
# clip_limit = 0.01
# nbins = 256
# outline_sigma = 1.0

def gpu_imshow(image_gpu):
  img = cp.asnumpy(image_gpu)
  plt.imshow(img)
  plt.axis('off')  # Turn off axis if needed
  plt.show()



def intersect_clahe_go(mask,image, kernel_size, clip_limit, nbins, outline_sigma):
    mask = cp.asarray(tf.imread(mask))
    image = cp.asarray(tf.imread(image))
    image[mask == 0] = 0    
    image_clahe = exposure.equalize_adapthist(cp.asnumpy(image), kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    #isinstance(image_clahe, np.ndarray)
    image_gol = gaussian(cp.asarray(image_clahe), sigma=outline_sigma, preserve_range=True)
    threshold = threshold_otsu(image_gol)
    binary = image_gol >= threshold
    image_fh = morphology.remove_small_holes(binary, area_threshold=10000)
    label_image = label(image_fh)
    #gpu_imshow(label2rgb(label_image))
    label_image = cp.asnumpy(label_image)
    label_image.dtype
    return label_image


for idx, (mask_file, intensity_file) in enumerate(zip(mask_files, intensity_files), start=1):
    print(f"Processing {idx} of {len(intensity_files)}")
    image_gol = intersect_clahe_go(mask_file, intensity_file, args.kernel_size, args.clip_limit, args.nbins, args.outline_sigma)
    output_path = os.path.join(input_folder, f"{os.path.basename(intensity_file)[:-4]}_reseg_clahe.tif")
    tf.imwrite(output_path, image_gol, compression='zlib')









