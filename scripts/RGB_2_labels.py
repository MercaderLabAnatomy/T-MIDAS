import os
import numpy as np
from skimage.io import imread
from tifffile import imwrite
from tqdm import tqdm
import argparse
import concurrent.futures

def parse_args():
    parser = argparse.ArgumentParser(description="Convert RGB images to label images.")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing RGB images in .tif format.")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="Number of threads to use for processing.")
    return parser.parse_args()

def rgb_to_labels(file, color_map):
    try:
        img = imread(file)
        img = np.array(img)
        img_labels = np.argmin(np.linalg.norm(color_map - img[:, :, None], axis=3), axis=2)
        return img_labels, None
    except Exception as e:
        return None, str(e)

def process_file(file, folder, color_map):
    input_path = os.path.join(folder, file)
    output_path = os.path.join(folder, file.replace(".tif", "_labels.tif"))
    
    img_labels, error = rgb_to_labels(input_path, color_map)
    
    if error:
        return f"Error processing {file}: {error}"
    
    try:
        imwrite(output_path, img_labels, compression='lzw')
        return f"Successfully processed {file}"
    except Exception as e:
        return f"Error writing {file}: {str(e)}"

def main():
    args = parse_args()
    folder = args.folder
    threads = args.threads

    # Define the color mapping
    color_map = np.array([[0, 0, 255],    # Blue
                          [0, 255, 0],    # Green
                          [255, 0, 0]])   # Red

    # Get all the .tif files in the folder
    files = [f for f in os.listdir(folder) if f.endswith(".tif") and not f.endswith("_labels.tif")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_file = {executor.submit(process_file, file, folder, color_map): file for file in files}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files), desc="Processing files"):
            file = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{file} generated an exception: {exc}')

if __name__ == "__main__":
    main()
