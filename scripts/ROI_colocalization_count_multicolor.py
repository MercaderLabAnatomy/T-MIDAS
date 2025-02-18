import os
import glob
import csv
from skimage import io
import argparse
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    parser.add_argument('--get_areas', type=str, default='n', help='Do you want to get areas of ROIs in all channels? (y/n)')
    parser.add_argument('--area_method', type=str, choices=['average', 'sum'], help='Method to calculate areas for second and third channels: "average" or "sum" (only used if get_areas is "y")')
    
    args = parser.parse_args()
    
    if args.get_areas.lower() == 'y' and args.area_method is None:
        args.area_method = input("\nWhich area stats? Type average or sum: ")
        while args.area_method not in ['average', 'sum']:
            print("Invalid input. Please enter 'average' or 'sum'.")
            args.area_method = input("\nWhich area stats? Type average or sum: ")
    
    return args

# In your main function:
args = parse_args()


def load_image(file_path):
    try:
        return io.imread(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None

def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

def safe_std(arr):
    return np.std(arr) if len(arr) > 0 else 0

def safe_sum(arr):
    return np.sum(arr) if len(arr) > 0 else 0

def coloc_channels(file_lists, channels, get_areas, area_method=None):
    csv_rows = []

    file_paths = file_lists[channels[0]]

    for file_path in tqdm(file_paths, total=len(file_paths), desc="Processing images"):
        try:
            images = [load_image(file_lists[channel][file_paths.index(file_path)]) for channel in channels]
            if any(img is None for img in images):
                continue

            image_c1, image_c2 = images[:2]
            image_c3 = images[2] if len(channels) == 3 else None

            label_ids = np.unique(image_c1)
            label_ids = label_ids[label_ids != 0]

            for label_id in label_ids:
                ROI_mask = image_c1 == label_id
                c2_in_c1_count = len(np.unique(image_c2 * ROI_mask)) - 1
                c3_in_c1_count = len(np.unique(image_c3 * ROI_mask)) - 1 if image_c3 is not None else 0

                if image_c3 is not None:
                    c3_in_c2_in_c1_count = len(np.unique(image_c3 * (image_c2 * ROI_mask))) - 1
                    c3_not_in_c2_but_in_c1_count = len(np.unique(image_c3 * (ROI_mask & ~image_c2))) - 1

                if get_areas.lower() == 'y':
                    props = regionprops(ROI_mask.astype(np.int32))
                    area = props[0].area if props else 0
                    c2_in_c1_areas = [prop.area for prop in regionprops(label(image_c2 * ROI_mask).astype(np.int32))]

                    if area_method == 'average':
                        c2_in_c1_area_value = safe_mean(c2_in_c1_areas)
                        c2_in_c1_std_area = safe_std(c2_in_c1_areas)
                    else:  # sum
                        c2_in_c1_area_value = safe_sum(c2_in_c1_areas)
                        c2_in_c1_std_area = None

                    if image_c3 is not None:
                        c3_in_c1_areas = [prop.area for prop in regionprops(label(image_c3 * ROI_mask).astype(np.int32))]
                        c3_in_c2_in_c1_areas = [prop.area for prop in regionprops(label(image_c3 * (image_c2 * ROI_mask)).astype(np.int32))]

                        if area_method == 'average':
                            c3_in_c1_area_value = safe_mean(c3_in_c1_areas)
                            c3_in_c2_in_c1_area_value = safe_mean(c3_in_c2_in_c1_areas)
                            c3_in_c1_std_area = safe_std(c3_in_c1_areas)
                            c3_in_c2_in_c1_std_area = safe_std(c3_in_c2_in_c1_areas)
                        else:  # sum
                            c3_in_c1_area_value = safe_sum(c3_in_c1_areas)
                            c3_in_c2_in_c1_area_value = safe_sum(c3_in_c2_in_c1_areas)
                            c3_in_c1_std_area = None
                            c3_in_c2_in_c1_std_area = None

                        row = [os.path.basename(file_path), label_id, area, c2_in_c1_count, c3_in_c1_count, c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count, 
                               c2_in_c1_area_value, c3_in_c1_area_value, c3_in_c2_in_c1_area_value]
                        if area_method == 'average':
                            row.extend([c2_in_c1_std_area, c3_in_c1_std_area, c3_in_c2_in_c1_std_area])
                        csv_rows.append(row)
                    else:
                        row = [os.path.basename(file_path), label_id, area, c2_in_c1_count, c2_in_c1_area_value]
                        if area_method == 'average':
                            row.append(c2_in_c1_std_area)
                        csv_rows.append(row)


        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    return csv_rows

def main():
    try:
        args = parse_args()
        parent_dir = args.input
        channels = args.channels
        label_patterns = args.label_patterns
        get_areas = args.get_areas
        area_method = args.area_method if get_areas.lower() == 'y' else None

        if len(set(channels)) < len(channels) or len(channels) < 2 or len(channels) > 3:
            raise ValueError("Channel names must be unique and 2 or 3 channels must be provided.")

        file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

        csv_rows = coloc_channels(file_lists, channels, get_areas, area_method)

        csv_file = os.path.join(parent_dir, 'colocalization.csv')

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Filename', f"{channels[0]}_ROI"]
            if get_areas.lower() == 'y':
                header.append(f"{channels[0]}_ROI_size")
            header.append(f"{channels[1]}_in_{channels[0]}")
            if get_areas.lower() == 'y':
                area_type = "avg" if area_method == "average" else "sum"
                header.append(f"{channels[1]}_{area_type}_size_in_{channels[0]}")
                if area_method == 'average':
                    header.append(f"{channels[1]}_std_size_in_{channels[0]}")
            if len(channels) == 3:
                header.extend([f"{channels[2]}_in_{channels[0]}", f"{channels[2]}_in_{channels[1]}_in_{channels[0]}", f"{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}"])
                if get_areas.lower() == 'y':
                    header.extend([f"{channels[2]}_{area_type}_size_in_{channels[0]}", 
                                   f"{channels[2]}_{area_type}_size_in_{channels[1]}_in_{channels[0]}"])
                    if area_method == 'average':
                        header.extend([f"{channels[2]}_std_size_in_{channels[0]}", 
                                       f"{channels[2]}_std_size_in_{channels[1]}_in_{channels[0]}"])
            writer.writerow(header)
            writer.writerows(csv_rows)

        print(f"Colocalization results saved to {csv_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")




if __name__ == "__main__":
    main()
