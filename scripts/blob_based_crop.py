import os
import tifffile as tf
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Blob-based crops.')
parser.add_argument('--input', type=str, help='path to the input folder containing intensity and label images.')
parser.add_argument('--blobfiles', type=str, help='tag of label images')
parser.add_argument('--intensityfiles', type=str, help='tag of intensity images')
parser.add_argument('--output_tag', type=str, help='tag of output images')
args = parser.parse_args()

# input = "/mnt/TEST/crop_mc"
# blobfiles = "_tissue_labels.tif"
# intensityfiles = "_nuclei_intensities.tif"

filenames = [f.replace(args.blobfiles, '') for f in os.listdir(args.input) if f.endswith(args.blobfiles)]
# os.path.join(input, filenames[0] + blobfiles)
# os.path.join(input, filenames[0] + intensityfiles)

# Iterate over all files in the directory
for filename in filenames:
    # Load the binary image and the original image
    binary_image = tf.imread(os.path.join(args.input, 
                                          filename + args.blobfiles))
    original_image = tf.imread(os.path.join(args.input, 
                                            filename + args.intensityfiles))
    # set each value in the original image to zero where the binary image is not zero
    original_image[binary_image == 0] = 0
    # save the cropped image
    tf.imwrite(os.path.join(args.input, 
                            filename + args.output_tag), 
               original_image, compression='zlib')