import os
import glob
import numpy as np
from skimage import io as skio
from tifffile import imwrite

def process_label_images_skimage_tiff(folder_path, selected_label_id, output_folder="processed_labels"):
    """
    Loads all images in a folder using skimage.io.imread, allows user to select a label ID,
    deletes all other labels, and exports the result as uint32 label images using tifffile with zlib compression.

    Args:
        folder_path (str): Path to the folder containing label images.
        selected_label_id (int): The label ID to keep in the images.
        output_folder (str, optional): Folder to save processed images. Defaults to "processed_labels".
    """

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving processed images to: {output_folder}")

    image_files = glob.glob(os.path.join(folder_path, '*.*'))  # Get all files in the folder
    label_image_files = []
    for file in image_files:
        try:
            # Try to load with skimage.io to check if it's a readable image
            img = skio.imread(file)
            if img.ndim >= 2: # Basic check if loaded as image
                label_image_files.append(file)
            else:
                print(f"Skipping file '{file}': Loaded data is not a valid image (dimensions < 2)")
        except Exception as e:
            print(f"Skipping file '{file}': Not a readable image or error loading ({e})")

    if not label_image_files:
        print(f"No readable images found in folder: {folder_path}")
        return

    print(f"Found {len(label_image_files)} images in folder.")

    for file_path in label_image_files:
        try:
            print(f"Processing: {file_path}")
            label_image = skio.imread(file_path)

            # Create a new array with only the selected label and 0 for others
            processed_image = np.zeros_like(label_image, dtype=np.uint32)
            processed_image[label_image == selected_label_id] = selected_label_id

            # Construct output file name
            file_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(file_name)[0]
            output_file_path = os.path.join(output_folder, f"{name_without_ext}_label_{selected_label_id}.tif") # Save as TIFF

            imwrite(output_file_path, processed_image, compression='zlib')
            print(f"Saved processed image: {output_file_path}")

        except Exception as e:
            print(f"Error processing '{file_path}': {e}")

    print("Label image processing complete.")


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing label images: ")
    while True:
        try:
            selected_label_id = int(input("Enter the label ID to keep (integer): "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer for the label ID.")

    process_label_images_skimage_tiff(folder_path, selected_label_id)