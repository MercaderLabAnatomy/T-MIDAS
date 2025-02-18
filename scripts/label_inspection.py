import os
import napari
from skimage.io import imread
import argparse
from tifffile import imwrite
import numpy as np
from tqdm import tqdm


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
    image_uint32 = image.astype(np.uint32)
    imwrite(filename, image_uint32, compression='zlib')

def load_and_edit_labels(folder_path, label_suffix, intensity=False):
    """
    Load label images from a folder and edit them using napari.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter files based on the label suffix
    label_files = [file for file in files if file.endswith(label_suffix)]

    
    # Load and edit each label file (add tqdm for progress bar)
    for file in tqdm(label_files, desc="Processing label images"):
        file_path = os.path.join(folder_path, file)
        label_image = imread(file_path)

        if intensity == True:
            intensity_file = file.replace(label_suffix, '.tif')
            intensity_path = os.path.join(folder_path, intensity_file)
            image = imread(intensity_path)
        
        # Open napari viewer
        viewer = napari.Viewer()
        if intensity == True:
            viewer.add_image(image, name='Image')
        viewer.add_labels(label_image, name='Label Image')
    


        
        print(f"Napari viewer opened for {file}.")
        print("Once you are satisfied with the labels, close the Napari viewer window.")
        
        # Run napari event loop
        napari.run()
        # save the edited labels
        edited_labels = viewer.layers['Label Image'].data
        save_image(edited_labels, file_path)

        print(f"Napari viewer closed for {file}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing with napari.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    # ask user if they want to also add the intensity image
    parser.add_argument("--intensity", type=str, default=False, help="Also load intensity image?")
    return parser.parse_args()

def main():
    args = parse_args()
    load_and_edit_labels(args.input, args.suffix, str2bool(args.intensity))

if __name__ == "__main__":
    main()
