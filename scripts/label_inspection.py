import os
import napari
from skimage.io import imread
import argparse

def load_and_edit_labels(folder_path, label_suffix):
    """
    Load label images from a folder and edit them using napari.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter files based on the label suffix
    label_files = [file for file in files if file.endswith(label_suffix)]
    
    # Load and edit each label file
    for file in label_files:
        file_path = os.path.join(folder_path, file)
        label_image = imread(file_path)
        
        # Open napari viewer
        viewer = napari.Viewer()
        viewer.add_labels(label_image, name='Label Image')
        
        print(f"Napari viewer opened for {file}.")
        print("Once you are satisfied with the labels, close the Napari viewer window.")
        
        # Run napari event loop
        napari.run()
        
        print(f"Napari viewer closed for {file}.")

def parse_args():
    parser = argparse.ArgumentParser(description="Loads label images from a folder for inspection and editing with napari.")
    parser.add_argument("--input", type=str, required=True, help="Path to the folder containing label images.")
    parser.add_argument("--suffix", type=str, required=True, help="Suffix of the label images (e.g., _labels.tif).")
    return parser.parse_args()

def main():
    args = parse_args()
    load_and_edit_labels(args.input, args.suffix)

if __name__ == "__main__":
    main()
