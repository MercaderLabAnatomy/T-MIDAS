import os
import glob
import argparse
import traceback
import csv
from skimage import io, measure
import numpy as np
from tqdm import tqdm
import concurrent.futures
import multiprocessing

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate region properties for labeled objects in images and save to CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing the images.')
    parser.add_argument('--label_pattern', type=str, required=True, help='File pattern for the label images (e.g., *_labels.tif).')
    parser.add_argument('--use_intensity', action='store_true', help='Use corresponding intensity images for region property calculation.')
    parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count(), help='Number of worker processes to use.')
    return parser.parse_args()

def process_label_image(args):
    """
    Processes a single label image to extract regionprops for each class,
    optionally using an intensity image.
    """
    label_image_path, intensity_image_path, include_intensity_props = args
    results = []
    try:
        label_image = io.imread(label_image_path)
        intensity_image = io.imread(intensity_image_path) if intensity_image_path else None

        if intensity_image is not None and intensity_image.shape != label_image.shape:
            raise ValueError("Intensity image must have the same shape as the label image.")

        class_labels = np.unique(label_image)
        class_labels = class_labels[class_labels != 0]  # Exclude background

        for class_label in class_labels:
            class_mask = (label_image == class_label)
            labeled_class = measure.label(class_mask, connectivity=1)
            regionprops = measure.regionprops(labeled_class, intensity_image=intensity_image)

            for prop in regionprops:
                if prop.label == 0:
                    continue
                row = [
                    os.path.basename(label_image_path),
                    int(class_label),
                    prop.label,
                    int(prop.area)
                ]
                if include_intensity_props and intensity_image is not None:
                    row.append(int(prop.mean_intensity))
                    row.append(int(prop.intensity_image.std()))
                results.append(row)

    except Exception as e:
        print(f"Error processing image {label_image_path}: {str(e)}")
        traceback.print_exc()
    return results

def main():
    """Main function to execute the region properties calculation and save to CSV."""
    try:
        args = parse_args()
        input_dir = args.input
        label_pattern = args.label_pattern
        use_intensity = args.use_intensity
        num_workers = args.num_workers

        label_image_paths = sorted(glob.glob(os.path.join(input_dir, label_pattern)))
        if not label_image_paths:
            raise FileNotFoundError(f"No label images found in {input_dir} matching pattern {label_pattern}")

        csv_filename = os.path.join(input_dir, "region_properties.csv")
        csv_header = ["Image", "Class", "Label", "Size"]
        if use_intensity:
            csv_header.append("Mean_Intensity")
            csv_header.append("Std_Intensity")

        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

            process_args = []
            for label_image_path in label_image_paths:
                base_name = os.path.splitext(os.path.basename(label_image_path))[0]
                base_name = base_name.replace(label_pattern.replace("*", "").replace(".tif", ""), "")
                intensity_image_path = os.path.join(input_dir, base_name + ".tif") if use_intensity else None
                
                if use_intensity and not os.path.exists(intensity_image_path):
                    print(f"Warning: Corresponding intensity image not found for {label_image_path}")
                    intensity_image_path = None

                process_args.append((label_image_path, intensity_image_path, use_intensity))

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_label_image, process_args), total=len(process_args), desc="Processing images"))

            for result in results:
                csv_writer.writerows(result)

        print(f"Region properties saved to {csv_filename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
