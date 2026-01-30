import os
import napari
from skimage.io import imread
import argparse
from tifffile import imwrite
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import gc
import warnings

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="data shape .* exceeds GL_MAX_TEXTURE_SIZE")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_image(image, filename):
    try:
        image_uint32 = image.astype(np.uint32)
        imwrite(filename, image_uint32, compression='zlib')
        return True
    except Exception as e:
        print(f"Error saving image to {filename}: {e}")
        return False

def get_image_shape(image_path):
    """Get image dimensions without loading the full image"""
    try:
        from tifffile import TiffFile
        with TiffFile(image_path) as tif:
            shape = tif.pages[0].shape
            return shape
    except Exception as e:
        print(f"Warning: Could not check image size for {image_path}: {e}")
        return None

def downscale_image(image, max_size=8192):
    """Downscale a large image to a more manageable size"""
    from skimage.transform import resize
    if max(image.shape) <= max_size:
        return image, 1.0
    factor = max_size / max(image.shape)
    new_shape = (int(image.shape[0] * factor), int(image.shape[1] * factor))
    print(f"Downscaling image from {image.shape} to {new_shape} for viewing (scale factor: {factor:.3f})")
    original_dtype = image.dtype
    resized = resize(image, new_shape, order=0, preserve_range=True, anti_aliasing=False)
    return resized.astype(original_dtype), factor

def upscale_edited_labels(edited_labels, original_shape, scale_factor):
    from skimage.transform import resize
    print(f"Upscaling edited labels from {edited_labels.shape} to {original_shape}")
    upscaled = resize(edited_labels, original_shape, order=0, preserve_range=True, anti_aliasing=False)
    return upscaled.astype(np.uint32)

def load_and_edit_labels(folder_path, label_suffix, intensity=False, max_size=8192, second_label_suffix=None):
    """
    Load label images from a folder and edit them using napari.
    Optionally also load and edit a second set of label images.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: The directory {folder_path} does not exist.")
        return

    try:
        files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error accessing directory {folder_path}: {e}")
        return

    label_files = [file for file in files if file.endswith(label_suffix)]
    if not label_files:
        print(f"No files with suffix '{label_suffix}' found in {folder_path}")
        return

    for file in tqdm(label_files, desc="Processing label images"):
        gc.collect()
        file_path = os.path.join(folder_path, file)
        original_shape = get_image_shape(file_path)
        if original_shape is None:
            print(f"Skipping {file} due to file reading issues")
            continue

        try:
            print(f"Loading label image {file_path}")
            label_image = imread(file_path)
            backup_path = file_path + ".backup"
            if not os.path.exists(backup_path):
                try:
                    save_image(label_image, backup_path)
                    print(f"Created backup at {backup_path}")
                except Exception as e:
                    print(f"Warning: Failed to create backup: {e}")
        except Exception as e:
            print(f"Error loading label image {file_path}: {e}")
            continue

        is_large = max(label_image.shape) > max_size
        scale_factor = 1.0
        original_shape = label_image.shape
        if is_large:
            label_image, scale_factor = downscale_image(label_image, max_size)
            print(f"Image downscaled for viewing. Full resolution will be preserved when saving.")

        # Load intensity image if requested
        image = None
        if intensity:
            try:
                intensity_file = file.replace(label_suffix, '.tif')
                intensity_path = os.path.join(folder_path, intensity_file)
                if os.path.exists(intensity_path):
                    print(f"Loading intensity image {intensity_path}")
                    intensity_image = imread(intensity_path)
                    if max(intensity_image.shape) > max_size:
                        image, _ = downscale_image(intensity_image, max_size)
                    else:
                        image = intensity_image
                else:
                    print(f"Warning: Intensity image {intensity_path} not found. Continuing with labels only.")
            except Exception as e:
                print(f"Error loading intensity image {intensity_path}: {e}")
                print("Continuing with labels only.")

        # Optionally load second label image
        second_label_image = None
        second_label_name = None
        if second_label_suffix:  # Only proceed if not None or empty
            second_label_file = file.replace(label_suffix, second_label_suffix)
            second_label_path = os.path.join(folder_path, second_label_file)
            if os.path.exists(second_label_path):
                print(f"Loading second label image {second_label_path}")
                second_label_image = imread(second_label_path)
                if max(second_label_image.shape) > max_size:
                    second_label_image, _ = downscale_image(second_label_image, max_size)
                second_label_name = second_label_file
            else:
                print(f"Second label image {second_label_path} not found. Skipping.")

        try:
            print("Starting napari viewer...")
            viewer = napari.Viewer()
            if intensity and image is not None:
                viewer.add_image(image, name='Image')
            viewer.add_labels(label_image, name=file)
            if second_label_image is not None:
                viewer.add_labels(second_label_image, name=second_label_name)
            print(f"Napari viewer opened for {file}.")
            print("Once you are satisfied with the labels, close the Napari viewer window.")
            napari.run()

            # Save first label
            if file in viewer.layers:
                edited_labels = viewer.layers[file].data
                if is_large:
                    print("Upscaling edited labels to original resolution...")
                    try:
                        edited_labels = upscale_edited_labels(edited_labels, original_shape, scale_factor)
                    except Exception as e:
                        print(f"Error during upscaling: {e}")
                        print("Saving at current resolution instead.")
                if save_image(edited_labels, file_path):
                    print(f"Successfully saved edited labels for {file}.")
                else:
                    print(f"Failed to save edited labels for {file}.")
            else:
                print(f"Warning: Layer '{file}' was removed or renamed. No changes saved.")

            # Save second label if edited
            if second_label_image is not None and second_label_name in viewer.layers:
                edited_second = viewer.layers[second_label_name].data
                second_label_path = os.path.join(folder_path, second_label_name)
                if is_large and edited_second.shape != original_shape:
                    try:
                        edited_second = upscale_edited_labels(edited_second, original_shape, scale_factor)
                    except Exception as e:
                        print(f"Error upscaling second label: {e}")
                if save_image(edited_second, second_label_path):
                    print(f"Successfully saved edited second label for {second_label_name}.")
                else:
                    print(f"Failed to save edited second label for {second_label_name}.")

            viewer.close()
            del viewer
            del label_image
            if image is not None:
                del image
            if second_label_image is not None:
                del second_label_image
            gc.collect()
            print(f"Napari viewer closed for {file}.")

        except KeyboardInterrupt:
            print("User interrupted the process. Exiting gracefully.")
            break
        except Exception as e:
            print(f"Error during napari session for {file}: {e}")
            continue

def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing with napari.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    parser.add_argument("--second_suffix", type=str, default=None, help="(Optional) Suffix for a second label image (e.g., _labels2.tif).")
    parser.add_argument("--intensity", type=str, default="False", help="Also load intensity image? (True/False)")
    parser.add_argument("--max-size", type=int, default=8192, help="Maximum dimension for viewing (default: 8192)")
    return parser.parse_args()

def main():
    try:
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        args = parse_args()
        load_and_edit_labels(
            args.input,
            args.suffix,
            str2bool(args.intensity),
            args.max_size,
            second_label_suffix=args.second_suffix
        )
        print("Program completed successfully.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
