import os
import glob
import csv
from skimage import io
import argparse
import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm

"""
This script is used to calculate the colocalization of ROIs in two or three color channels based on Napari label images.

It calculates the number of ROIs in the first channel that have pixels in the second and third channels.
It also calculates the number of ROIs in the first channel that have pixels in the second channel but not in the third channel.
The script can also calculate the areas of ROIs in the first channel.
The results are saved in a CSV file.


Description: This script calculates the colocalization of ROIs in two or three color channels based on Napari label images.
A Napari label id is considered as a region of interest (ROI). Label images of each channel are to be provided in separate folders.

The following colocalizations counts are available:
- c2_in_c1_count: Number of ROIs in the second channel that are inside ROIs in the first channel.
- c3_in_c1_count: Number of ROIs in the third channel that are inside ROIs in the first channel.
- c3_in_c2_in_c1_count: Number of ROIs in the third channel that are inside ROIs in the second channel 
  that are inside ROIs in the first channel.
- c3_not_in_c2_but_in_c1_count: Number of ROIs in the third channel that are inside ROIs in the first channel 
  but not inside ROIs in the second channel.

Colocalization is performed by Boolean masking of the ROIs in the second and third channels with the ROIs in the first channel. 
Counts are then obtained by calculating the length of the vector of unique labels in the masked label images.

The script can also calculate the size of ROIs in the first channel.

The results are saved in a CSV file.


"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    # parser.add_argument('--output_images', type=str, default='n', help='Do you want to save colocalization images? (y/n)')
    parser.add_argument('--get_areas', type=str, default='n', help='Do you want to get areas of ROIs in the first channel? (y/n)')
    return parser.parse_args()

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

def coloc_channels(file_lists, channels, get_areas):
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

                # if output_images.lower() == 'y':
                #     coloc_image_c2 = label(ROI_mask & (image_c2 > 0))
                #     filename = os.path.splitext(os.path.basename(file_path))[0]
                #     io.imsave(f"{filename}_{channels[1]}_in_{channels[0]}_ROI_{label_id}.tif", coloc_image_c2, compression='zlib')
                #     if image_c3 is not None:
                #         coloc_image_c3 = label(ROI_mask & (image_c3 > 0))
                #         io.imsave(f"{filename}_{channels[2]}_in_{channels[0]}_ROI_{label_id}.tif", coloc_image_c3, compression='zlib')

                if get_areas.lower() == 'y':
                    props = regionprops(ROI_mask.astype(np.int32))
                    area = props[0].area if props else 0
                    c2_in_c1_areas = [prop.area for prop in regionprops(label(image_c2 * ROI_mask).astype(np.int32))] # this is
                    c2_in_c1_avg_area = safe_mean(c2_in_c1_areas)
                    c2_in_c1_std_area = safe_std(c2_in_c1_areas)

                    if image_c3 is not None:
                        c3_in_c1_areas = [prop.area for prop in regionprops(label(image_c3 * ROI_mask).astype(np.int32))]
                        c3_in_c1_avg_area = safe_mean(c3_in_c1_areas)
                        c3_in_c1_std_area = safe_std(c3_in_c1_areas)
                        c3_in_c2_in_c1_areas = [prop.area for prop in regionprops(label(image_c3 * (image_c2 * ROI_mask)).astype(np.int32))]
                        c3_in_c2_in_c1_avg_area = safe_mean(c3_in_c2_in_c1_areas)
                        c3_in_c2_in_c1_std_area = safe_std(c3_in_c2_in_c1_areas)
                        csv_rows.append([os.path.basename(file_path), label_id, area, c2_in_c1_count, c3_in_c1_count, c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count, 
                                         c2_in_c1_avg_area, c2_in_c1_std_area, 
                                         c3_in_c1_avg_area, c3_in_c1_std_area,
                                         c3_in_c2_in_c1_avg_area, c3_in_c2_in_c1_std_area])
                    else:
                        csv_rows.append([os.path.basename(file_path), label_id, area, c2_in_c1_count, c2_in_c1_avg_area, c2_in_c1_std_area])
                else:
                    if image_c3 is not None:
                        csv_rows.append([os.path.basename(file_path), label_id, c2_in_c1_count, c3_in_c1_count,c3_in_c2_in_c1_count])
                    else:
                        csv_rows.append([os.path.basename(file_path), label_id, c2_in_c1_count])

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    return csv_rows

def main():
    try:
        args = parse_args()
        parent_dir = args.input
        #channels = [c.upper() for c in args.channels] # actually , we don't need to convert to upper case
        channels = args.channels
        label_patterns = args.label_patterns
        # output_images = args.output_images
        get_areas = args.get_areas

        if len(set(channels)) < len(channels) or len(channels) < 2 or len(channels) > 3:
            raise ValueError("Channel names must be unique and 2 or 3 channels must be provided.")

        file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

        csv_rows = coloc_channels(file_lists, channels, get_areas)

        csv_file = os.path.join(parent_dir, 'colocalization.csv')

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Filename', f"{channels[0]}_ROI"]
            if get_areas.lower() == 'y':
                header.append(f"{channels[0]}_ROI_size")
            header.extend([f"{channels[1]}_in_{channels[0]}"])
            if len(channels) == 3:
                header.extend([f"{channels[2]}_in_{channels[0]}", f"{channels[2]}_in_{channels[1]}_in_{channels[0]}", f"{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}"])
            if get_areas.lower() == 'y':
                header.extend([f"{channels[1]}_avg_size_in_{channels[0]}", f"{channels[1]}_std_size_in_{channels[0]}"])
                if len(channels) == 3:
                    header.extend([f"{channels[2]}_avg_size_in_{channels[0]}", f"{channels[2]}_std_size_in_{channels[0]}", f"{channels[2]}_avg_size_in_{channels[1]}_in_{channels[0]}", f"{channels[2]}_std_size_in_{channels[1]}_in_{channels[0]}"])
            writer.writerow(header)
            writer.writerows(csv_rows)

        print(f"Colocalization results saved to {csv_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
