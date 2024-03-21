import os
import glob
import tifffile as tf
import argparse
from skimage import exposure
import cupy as cp
from cucim.skimage import morphology
from cucim.skimage.filters import gaussian
from cucim.skimage.filters.thresholding import threshold_otsu
from cucim.skimage.measure import label

# Argument Parsing
parser = argparse.ArgumentParser(description="Segments CLAHE images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--masks", type=str, required=True, help="Path to label images.")
parser.add_argument('--label_pattern', type=str, help='Label image suffix. Example: "*_labels.tif"')
parser.add_argument("--kernel_size", type=int, required=True, help="Defines the shape of contextual regions.")
parser.add_argument("--clip_limit", type=float, required=True, help="Defines the contrast limit for localised histogram equalisation.")
parser.add_argument("--nbins", type=int, required=True, help="Number of bins for the histogram.")
parser.add_argument("--outline_sigma", type=float, default=1.0, help="Defines the sigma for the gauss-otsu-labeling.")
args = parser.parse_args()

input_folder = args.input
label_pattern = args.label_pattern

intensity_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif') and not f.endswith('_labels.tif')]
#mask_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(label_pattern)]
mask_files = glob.glob(os.path.join(input_folder, label_pattern))

mask_files.sort()
intensity_files.sort()

# compare number of files
if len(mask_files) != len(intensity_files):
    print(f"Number of mask files: {len(mask_files)}\n")
    print(f"Number of intensity files: {len(intensity_files)}\n")
    raise ValueError("Number of mask files and intensity files do not match.")

# check if filenames contain a small matching pattern in the middle of the filename
for mask_file, intensity_file in zip(mask_files, intensity_files):
    mask_name = os.path.basename(mask_file)
    intensity_name = os.path.basename(intensity_file)
    if mask_name.split('-')[1] != intensity_name.split('-')[1]: # 
        raise ValueError("Mask and intensity files do not match.")

def intersect_clahe_go(mask,image, kernel_size, clip_limit, nbins, outline_sigma):
    mask = cp.asarray(tf.imread(mask))
    image = cp.asarray(tf.imread(image))
    image[mask == 0] = 0    
    image_clahe = exposure.equalize_adapthist(cp.asnumpy(image), kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    image_gol = gaussian(cp.asarray(image_clahe), sigma=outline_sigma, preserve_range=True)
    threshold = threshold_otsu(image_gol)
    binary = image_gol >= threshold
    image_fh = morphology.remove_small_holes(binary, area_threshold=10000)
    label_image = label(image_fh)
    label_image = cp.asnumpy(label_image)
    return label_image

for idx, (mask_file, intensity_file) in enumerate(zip(mask_files, intensity_files), start=1):
    print(f"Processing {idx} of {len(intensity_files)}")
    image_gol = intersect_clahe_go(mask_file, intensity_file, args.kernel_size, args.clip_limit, args.nbins, args.outline_sigma)
    output_path = os.path.join(input_folder, f"{os.path.basename(intensity_file)[:-4]}_reseg_clahe.tif")
    tf.imwrite(output_path, image_gol, compression='zlib')









