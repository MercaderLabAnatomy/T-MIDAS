import os
from skimage import exposure, util
import tifffile as tf
import argparse



# Argument Parsing
parser = argparse.ArgumentParser(description="Applies CLAHE to image with intensity gradient")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--kernel_size", type=int, required=True, help="Defines the shape of contextual regions.")
parser.add_argument("--clip_limit", type=float, required=True, help="Defines the contrast limit for localised histogram equalisation.")
parser.add_argument("--nbins", type=int, required=True, help="Number of bins for the histogram.")
args = parser.parse_args()

def main():
    """Main function to process all images in the input directory."""
    image_folder = os.path.join(args.input)
    for filename in os.listdir(image_folder):
        if not filename.endswith(".tif"):
            continue
        # also exclude images that end with _labels.tif
        if filename.endswith("_labels.tif"):
            continue
        print(f"Processing image: {filename}")
        image_path = os.path.join(image_folder, filename)
        image = tf.imread(image_path)
        image_clahe = exposure.equalize_adapthist(image, kernel_size=args.kernel_size, clip_limit=args.clip_limit, nbins=args.nbins)
        if image_clahe is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_clahe_" + 
                           "ksize-" + str(args.kernel_size) +
                            "_cliplim-" + str(args.clip_limit) +
                            "_nbins-" + str(args.nbins) + ".tif")

            tf.imwrite(output_path, util.img_as_ubyte(image_clahe), compression='zlib')

if __name__ == "__main__":
    main()


