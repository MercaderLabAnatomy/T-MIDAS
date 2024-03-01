import os
import tifffile as tf
import pyclesperanto_prototype as cle  # version 0.24.2
import argparse



def tuple_of_floats(arg):
    return tuple(map(float, arg.split(',')))


# Argument Parsing
parser = argparse.ArgumentParser(description="Segments CLAHE images.")
parser.add_argument("--input", type=str, required=True, help="Path to input images.")
parser.add_argument("--min_box", type=tuple_of_floats, required=True, help="Defines the pixel cube neighborhood.")
parser.add_argument("--outline_sigma", type=float, default=1.0, help="Defines the sigma for the gauss-otsu-labeling.")
args = parser.parse_args()



def segment(image, min_box, outline_sigma):

    image_GPU = cle.push(image)
    image_mb = cle.minimum_box(image_GPU, None, min_box[0], min_box[1], min_box[2])
    image_gol = cle.gauss_otsu_labeling(image_mb, None, outline_sigma)
    image_gol = cle.pull(image_gol)
    return image_gol



def main():
    """Main function to process all images in the input directory."""
    image_folder = os.path.join(args.input)
    for filename in os.listdir(image_folder):
        if not filename.endswith(".tif"):
            continue
        print(f"Processing image: {filename}")
        image_path = os.path.join(image_folder, filename)
        image = tf.imread(image_path)
        label_image = segment(image, args.min_box, args.outline_sigma)
        if label_image is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_seg_clahe_" + 
                                       f"min_box-{args.min_box[0]}_{args.min_box[1]}_{args.min_box[2]}_outsig-{args.outline_sigma}.tif")

            tf.imwrite(output_path, label_image, compression='zlib')

if __name__ == "__main__":
    main()

