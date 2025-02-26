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
    
    # Don't resize if already smaller than max_size
    if max(image.shape) <= max_size:
        return image, 1.0
    
    # Calculate resize factor to fit within max_size
    factor = max_size / max(image.shape)
    new_shape = (int(image.shape[0] * factor), int(image.shape[1] * factor))
    
    print(f"Downscaling image from {image.shape} to {new_shape} for viewing (scale factor: {factor:.3f})")
    
    # Preserve image dtype
    original_dtype = image.dtype
    
    # For label images, use nearest neighbor to preserve label values
    resized = resize(image, new_shape, order=0, preserve_range=True, anti_aliasing=False)
    
    return resized.astype(original_dtype), factor

def upscale_edited_labels(edited_labels, original_shape, scale_factor):
    """Upscale the edited labels back to the original shape"""
    from skimage.transform import resize
    
    print(f"Upscaling edited labels from {edited_labels.shape} to {original_shape}")
    
    # Use nearest neighbor interpolation to preserve label values
    upscaled = resize(edited_labels, original_shape, order=0, 
                      preserve_range=True, anti_aliasing=False)
    
    return upscaled.astype(np.uint32)

def load_and_edit_labels(folder_path, label_suffix, intensity=False, max_size=8192):
    """
    Load label images from a folder and edit them using napari.
    Scale large images down for viewing and up for saving.
    """
    # Ensure folder path exists
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: The directory {folder_path} does not exist.")
        return

    # List all files in the folder
    try:
        files = os.listdir(folder_path)
    except Exception as e:
        print(f"Error accessing directory {folder_path}: {e}")
        return

    # Filter files based on the label suffix
    label_files = [file for file in files if file.endswith(label_suffix)]
    
    if not label_files:
        print(f"No files with suffix '{label_suffix}' found in {folder_path}")
        return
    
    # Process each label file
    for file in tqdm(label_files, desc="Processing label images"):
        # Force garbage collection between files
        gc.collect()
        
        file_path = os.path.join(folder_path, file)
        
        # Get original image shape
        original_shape = get_image_shape(file_path)
        if original_shape is None:
            print(f"Skipping {file} due to file reading issues")
            continue
            
        # Load label image with error handling
        try:
            print(f"Loading label image {file_path}")
            label_image = imread(file_path)
            
            # Create a backup of the original label image
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
        
        # Check if downscaling is needed
        is_large = max(label_image.shape) > max_size
        scale_factor = 1.0
        original_shape = label_image.shape
        
        # Downscale if needed
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
                    
                    # Downscale intensity image if needed
                    if max(intensity_image.shape) > max_size:
                        image, _ = downscale_image(intensity_image, max_size)
                    else:
                        image = intensity_image
                else:
                    print(f"Warning: Intensity image {intensity_path} not found. Continuing with labels only.")
            except Exception as e:
                print(f"Error loading intensity image {intensity_path}: {e}")
                print("Continuing with labels only.")
        
        # Open napari viewer
        try:
            print("Starting napari viewer...")
            viewer = napari.Viewer()
            
            if intensity and image is not None:
                viewer.add_image(image, name='Image')
            
            viewer.add_labels(label_image, name=file)
        
            print(f"Napari viewer opened for {file}.")
            print("Once you are satisfied with the labels, close the Napari viewer window.")
            
            # Run napari event loop
            napari.run()
            
            # Check if the layer still exists after viewer is closed
            if file in viewer.layers:
                # Get the edited labels
                edited_labels = viewer.layers[file].data
                
                # If we were working with downscaled data, upscale before saving
                if is_large:
                    print("Upscaling edited labels to original resolution...")
                    try:
                        edited_labels = upscale_edited_labels(edited_labels, original_shape, scale_factor)
                    except Exception as e:
                        print(f"Error during upscaling: {e}")
                        print("Saving at current resolution instead.")
                
                # Save the edited labels
                if save_image(edited_labels, file_path):
                    print(f"Successfully saved edited labels for {file}.")
                else:
                    print(f"Failed to save edited labels for {file}.")
            else:
                print(f"Warning: Layer '{file}' was removed or renamed. No changes saved.")
            
            # Clean up
            viewer.close()
            del viewer
            del label_image
            if image is not None:
                del image
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
    parser.add_argument("--intensity", type=str, default="False", help="Also load intensity image? (True/False)")
    parser.add_argument("--max-size", type=int, default=8192, help="Maximum dimension for viewing (default: 8192)")
    return parser.parse_args()

def main():
    try:
        # Set maximum numpy thread count to avoid resource exhaustion
        import os
        os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads
        os.environ["NUMEXPR_NUM_THREADS"] = "4"  # Limit numexpr threads
        os.environ["MKL_NUM_THREADS"] = "4"  # Limit MKL threads
        
        args = parse_args()
        load_and_edit_labels(args.input, args.suffix, str2bool(args.intensity), args.max_size)
        print("Program completed successfully.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()