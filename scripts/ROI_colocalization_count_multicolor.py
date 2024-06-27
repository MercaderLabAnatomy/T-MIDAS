import os
import glob
import csv
from skimage import io
import argparse
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cucim.skimage.measure import label, regionprops
import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True, help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    parser.add_argument('--output_images', type=str, default='n', help='Do you want to save colocalization images? (y/n)')
    parser.add_argument('--get_areas', type=str, default='n', help='Do you want to get areas of ROIs in the first channel? (y/n)')
    return parser.parse_args()

def coloc_channels(file_lists, channels, output_images, get_areas):
    csv_rows = []

    # Create a list of file paths for the first channel
    file_paths = file_lists[channels[0]]

    # Wrap the loop with tqdm
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Processing images"):

        image_c1 = imread(file_lists[channels[0]][file_paths.index(file_path)])
        image_c2 = imread(file_lists[channels[1]][file_paths.index(file_path)])
        if len(channels) == 3:
            image_c3 = imread(file_lists[channels[2]][file_paths.index(file_path)])
        else:
            pass

        label_ids = np.unique(image_c1)
        label_ids = label_ids[label_ids != 0] # drop the background label

        for label_id in label_ids:
            ROI_mask = image_c1 == label_id # boolean mask with true where label_id is present in image_c1
            c2_in_c1_count = len(np.unique(image_c2 * ROI_mask)) - 1

            #c2_not_in_c1_count = len(np.unique(image_c2 * ~ROI_mask)) - 1 # the tilde operator inverts the mask

            if len(channels) ==3:
                c3_in_c2_in_c1_count = len(np.unique(image_c3 * (image_c2 * ROI_mask))) - 1
                c3_not_in_c2_but_in_c1_count = len(np.unique(image_c3 * (ROI_mask & ~image_c2))) - 1 # this is the c3 counts that are in the ROI but not in c2

            if output_images.lower() == 'y': 
                coloc_image_c2 = label(ROI_mask & (image_c2 > 0))
                filename = os.path.splitext(os.path.basename(file_path))[0]
                imwrite(f"{filename}_{channels[1]}_in_{channels[0]}_ROI_{label_id}.tif", coloc_image_c2, compression='zlib')
                if len(channels) ==3:
                    coloc_image_c3 = label(ROI_mask & (image_c3 > 0))
                    imwrite(f"{filename}_{channels[2]}_in_{channels[0]}_ROI_{label_id}.tif", coloc_image_c3, compression='zlib')
                else:
                    pass


            if get_areas.lower() == 'y':
                area = regionprops(ROI_mask.astype(np.int32))[0].area
                if len(channels) == 2:
                    csv_rows.append([os.path.basename(file_path), label_id, area, c2_in_c1_count])
                elif len(channels) ==3:
                    csv_rows.append([os.path.basename(file_path), label_id, area, c2_in_c1_count, c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count])
                else:
                    raise ValueError("Number of channels must be 2 or 3.")
            else:
                if len(channels) == 2:
                    csv_rows.append([os.path.basename(file_path), label_id, c2_in_c1_count])
                elif len(channels) ==3:
                    csv_rows.append([os.path.basename(file_path), label_id, c2_in_c1_count, c3_in_c2_in_c1_count])
                else:
                    raise ValueError("Number of channels must be 2 or 3.")

    return csv_rows




def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    label_patterns = args.label_patterns
    output_images = args.output_images
    get_areas = args.get_areas

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

    csv_rows = coloc_channels(file_lists, channels, output_images, get_areas)

    csv_file = os.path.join(parent_dir, 'colocalization.csv')

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['Filename', f"{channels[0]} ROI"]
        if get_areas.lower() == 'y':
            header.append(f"{channels[0]} ROI size")
        if len(channels) == 2:
            header.append(f"{channels[1]}_in_{channels[0]}")
        elif len(channels) ==3:
            header.append(f"{channels[1]}_in_{channels[0]}")
            header.append(f"{channels[2]}_in_{channels[1]}_in_{channels[0]}")
            header.append(f"{channels[2]}_not_in_{channels[1]}_but_in_{channels[0]}")
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"Colocalization results saved to {csv_file}")

if __name__ == "__main__":
    main()
