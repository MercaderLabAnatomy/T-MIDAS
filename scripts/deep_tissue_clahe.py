import os
import numpy as np
from skimage import exposure, util
import tifffile as tf
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Applies CLAHE to 3D time series images with intensity gradient")
    parser.add_argument("--input", type=str, required=True, help="Path to input images.")
    parser.add_argument("--kernel_size", type=int, required=True, help="Defines the shape of contextual regions.")
    parser.add_argument("--clip_limit", type=float, required=True, help="Defines the contrast limit for localised histogram equalisation.")
    parser.add_argument("--nbins", type=int, required=True, help="Number of bins for the histogram.")
    parser.add_argument("--dim_order", type=str, default='TZCYX', help="Dimension order of the input images.")
    return parser.parse_args()

args = parse_args()

def apply_clahe_3d(image, kernel_size, clip_limit, nbins):
    """Apply CLAHE to each 2D slice of a 3D volume."""
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        result[i] = exposure.equalize_adapthist(image[i], kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    return result

def process_image(image, dim_order):
    """Process the image based on its dimension order."""
    # Identify the axes
    t_axis = dim_order.index('T') if 'T' in dim_order else None
    z_axis = dim_order.index('Z') if 'Z' in dim_order else None
    c_axis = dim_order.index('C') if 'C' in dim_order else None
    y_axis = dim_order.index('Y')
    x_axis = dim_order.index('X')

    # Rearrange axes to TZCYX order
    transpose_order = [t_axis, z_axis, c_axis, y_axis, x_axis]
    transpose_order = [i for i in transpose_order if i is not None]
    image = np.transpose(image, transpose_order)

    # Apply CLAHE
    if t_axis is not None and z_axis is not None:
        # 5D: TZCYX
        result = np.zeros_like(image, dtype=np.float32)
        for t in range(image.shape[0]):
            for c in range(image.shape[2]):
                result[t, :, c] = apply_clahe_3d(image[t, :, c], args.kernel_size, args.clip_limit, args.nbins)
    elif t_axis is not None:
        # 4D: TCYX
        result = np.zeros_like(image, dtype=np.float32)
        for t in range(image.shape[0]):
            for c in range(image.shape[1]):
                result[t, c] = exposure.equalize_adapthist(image[t, c], kernel_size=args.kernel_size, clip_limit=args.clip_limit, nbins=args.nbins)
    elif z_axis is not None:
        # 4D: ZCYX
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[1]):
            result[:, c] = apply_clahe_3d(image[:, c], args.kernel_size, args.clip_limit, args.nbins)
    else:
        # 3D: CYX
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            result[c] = exposure.equalize_adapthist(image[c], kernel_size=args.kernel_size, clip_limit=args.clip_limit, nbins=args.nbins)

    # Transpose back to original dimension order
    result = np.transpose(result, np.argsort(transpose_order))
    return result

def main():
    """Main function to process all images in the input directory."""
    image_folder = os.path.join(args.input)
    for filename in tqdm(os.listdir(image_folder), total=len(os.listdir(image_folder)), desc="Processing images"):
        if not filename.endswith(".tif") or filename.endswith("_labels.tif"):
            continue
        print(f"Processing image: {filename}")
        image_path = os.path.join(image_folder, filename)
        image = tf.imread(image_path)
        
        image_clahe = process_image(image, args.dim_order)
        
        if image_clahe is not None:
            output_path = os.path.join(image_folder, f"{filename[:-4]}_clahe_" + 
                           "ksize-" + str(args.kernel_size) +
                            "_cliplim-" + str(args.clip_limit) +
                            "_nbins-" + str(args.nbins) + ".tif")

            tf.imwrite(output_path, util.img_as_ubyte(image_clahe), compression='zlib')

if __name__ == "__main__":
    main()
