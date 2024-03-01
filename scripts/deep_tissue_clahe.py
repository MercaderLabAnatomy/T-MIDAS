import os
import numpy as np
from skimage import exposure
import mclahe as mc
import tifffile as tf
import argparse



# Argument Parsing
parser = argparse.ArgumentParser(description="Applies CLAHE to image with intensity gradient")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--kernel_size", type=int, default=64, help="Defines the size of the kernel for localised histogram equalisation.")
parser.add_argument("--clip_limit", type=float, default=0.01, help="Defines the clip limit for the histogram.")
parser.add_argument("--nbins", type=int, default=256, help="Number of bins for the histogram.")
parser.add_argument("--multicolor", type=bool, required=True, help="Are the images multicolor? (True/False)")

args = parser.parse_args()

# check if args.multicolor is y or n
if args.multicolor == "y":
    args.multicolor = True
else:
    args.multicolor = False
    

def main():
    """Main function to process all images in the input directory."""
    image_folder = os.path.join(args.input)
    for filename in os.listdir(image_folder):
        if not filename.endswith(".tif"):
            continue

        image_path = os.path.join(image_folder, filename)
        image = tf.imread(image_path)
        if args.multicolor:
            image_clahe = mc.mclahe(image, 
                                    kernel_size=args.kernel_size, 
                                    clip_limit=args.clip_limit, 
                                    nbins=args.nbins, 
                                    adaptive_hist_range=False, 
                                    use_gpu=False)
        else:
            image_clahe = exposure.equalize_adapthist(image, 
                                                      kernel_size=args.kernel_size, 
                                                      clip_limit=args.clip_limit, 
                                                      nbins=args.nbins)
        if image_clahe is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_clahe_" + 
                                       "ksize-" + str(args.kernel_size) +
                                        "_cliplim-" + str(args.clip_limit) +
                                        "_nbins-" + str(args.nbins) + ".tif")
            tf.imwrite(output_path, image_clahe, compression='zlib')
            print(f"\nProcessed image: {filename}")

if __name__ == "__main__":
    main()


