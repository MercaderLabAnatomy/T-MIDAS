import os
import glob
import csv
import argparse
from tqdm import tqdm
import numpy as np

try:
    import cupy as cp
    import cucim.skimage as cusk
    from skimage import io, measure

    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    from skimage import io, measure

    GPU_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Script for colocalization analysis of images.')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--label_patterns', nargs='+', type=str, required=True,
                        help='Label pattern for each channel. Example: "*_labels.tif *_labels.tif *_labels.tif"')
    parser.add_argument('--get_sizes', type=str, default='n', help='Do you want to get sizes of ROIs in all channels? (y/n)')
    parser.add_argument('--size_method', type=str, choices=['median', 'sum'],
                        help='Method to calculate sizes for second and third channels: "median" or "sum" (only used if get_sizes is "y")')

    args = parser.parse_args()

    if args.get_sizes.lower() == 'y' and args.size_method is None:
        args.size_method = input("\nWhich size stats? Type median or sum: ")
        while args.size_method not in ['median', 'sum']:
            print("Invalid input. Please enter 'median' or 'sum'.")
            args.size_method = input("\nWhich size stats? Type median or sum: ")

    return args


def load_image(file_path):
    if GPU_AVAILABLE:
        image = io.imread(file_path)
        return cp.asarray(image)
    else:
        return io.imread(file_path)


def safe_median(arr):
    return cp.median(arr).get() if GPU_AVAILABLE and len(arr) > 0 else np.median(arr) if len(arr) > 0 else 0


def safe_sum(arr):
    return cp.sum(arr).get() if GPU_AVAILABLE and len(arr) > 0 else np.sum(arr).get() if len(arr) > 0 else 0


def coloc_counts(image_c1, image_c2, image_c3=None):
    xp = cp if GPU_AVAILABLE else np

    # 1. Create boolean masks for nonzero entries
    mask_c1 = image_c1 != 0
    mask_c2 = image_c2 != 0

    # 2. Find unique label IDs in each channel
    label_ids = xp.unique(image_c1[mask_c1])  # Nonzero values in c1
    label_ids = label_ids[label_ids != 0] #remove zero label

    # Convert the NumPy/CuPy array to Python integers:
    if GPU_AVAILABLE:
        label_ids = [int(x) for x in label_ids.get()] # Convert to Python integers if using GPU
    else:
        label_ids = [int(x) for x in label_ids] # Convert to Python integers


    # 3. Count unique c2 labels within c1 regions
    c2_in_c1_count = len(xp.unique(image_c2[mask_c1 & mask_c2]))
    # c2_in_c1_count = len(xp.intersect1d(label_ids_c1, label_ids_c2))




    results = {
        "c2_in_c1_count": c2_in_c1_count,
        "label_ids": label_ids
    }

    # 4. If a third channel is provided, calculate additional stats
    if image_c3 is not None:
        mask_c3 = image_c3 != 0
        label_ids_c3 = xp.unique(image_c3[mask_c3])
        label_ids_c3 = label_ids_c3[label_ids_c3 != 0] #remove zero label
        c3_in_c2_in_c1_count = len(xp.unique(image_c3[mask_c1 & mask_c2 & mask_c3])) if image_c3 is not None else 0
        c3_not_in_c2_but_in_c1_count = len(xp.unique(image_c3[mask_c1 & ~mask_c2 & mask_c3])) if image_c3 is not None else 0

        results["c3_in_c2_in_c1_count"] = c3_in_c2_in_c1_count
        results["c3_not_in_c2_but_in_c1_count"] = c3_not_in_c2_but_in_c1_count

    return results

      
      
def calculate_all_rois_size(image):
    xp = cp if GPU_AVAILABLE else np
    measure_func = cusk.measure if GPU_AVAILABLE else measure
    sizes = {}
    for prop in measure_func.regionprops(image.astype(xp.int32)):
        label = int(prop.label)  # Ensure label is a standard Python integer
        area_value = float(prop.area) if hasattr(prop.area, 'item') else prop.area
        # print(f"Found region with label: {label}, area: {area_value}, type: {type(prop.area)}")
        sizes[label] = int(area_value)  # Convert to Python integer
    # print(f"Completed size calculation, found {len(sizes)} regions with sizes: {sizes}")
    return sizes

def calculate_coloc_area(image_c1, image_c2, label_id, mask_c2 = None):
    xp = cp if GPU_AVAILABLE else np
    mask = (image_c1 == label_id)

    if mask_c2 is not None:
        if mask_c2:
            mask_c2 = image_c2 != 0 #c2 mask
            mask = mask & mask_c2
        else:
            mask_c2 = image_c2 == 0 #not c2 mask
            mask = mask & mask_c2
    masked_image = image_c2 * mask
    area = xp.count_nonzero(masked_image)
    return area

def coloc_channels(file_lists, channels, get_sizes, size_method=None):
    csv_rows = []
    file_paths = file_lists[channels[0]]
    xp = cp if GPU_AVAILABLE else np
    measure_func = cusk.measure if GPU_AVAILABLE else measure

    for i, file_path in enumerate(tqdm(file_paths, total=len(file_paths), desc="Processing images")):
        try:
            images = [load_image(file_lists[channel][i]) for channel in channels]
            if any(img is None for img in images):
                continue

            image_c1, image_c2 = images[:2]
            image_c3 = images[2] if len(channels) == 3 else None

            # Calculate colocalization counts using the efficient method
            coloc_results = coloc_counts(image_c1, image_c2, image_c3)

            label_ids = coloc_results["label_ids"]

            c2_in_c1_count = coloc_results["c2_in_c1_count"]
            if image_c3 is not None:
                c3_in_c2_in_c1_count = coloc_results["c3_in_c2_in_c1_count"]
                c3_not_in_c2_but_in_c1_count = coloc_results["c3_not_in_c2_but_in_c1_count"]

            # Pre-calculate sizes for each label in image_c1 only once
            image_c1_sizes = {}  # Initialize to prevent errors
            if get_sizes.lower() == 'y':
                # print(f"Calculating sizes for image_c1 with shape: {image_c1.shape}")
                image_c1_sizes = calculate_all_rois_size(image_c1)
                # print(f"Got image_c1_sizes: {image_c1_sizes}")

            for idx, label_id in enumerate(label_ids):
                label_id_int = int(label_id)  # Convert to Python integer
                # print(f"Processing label_id: {label_id} (type: {type(label_id)}), converted to {label_id_int} (type: {type(label_id_int)})")
                row = [os.path.basename(file_path), label_id_int, c2_in_c1_count]

                if get_sizes.lower() == 'y':
                    size = image_c1_sizes.get(label_id_int, 0)
                    # print(f"Retrieved size for label {label_id_int}: {size}")

                    c2_in_c1_area = 0  # Initialize
                    if image_c2 is not None:
                        c2_in_c1_area = calculate_coloc_area(image_c1, image_c2, label_id) #passing the correct number of arguments

                    row.extend([size, c2_in_c1_area])

                if image_c3 is not None:
                    row.extend([c3_in_c2_in_c1_count, c3_not_in_c2_but_in_c1_count])
                    if get_sizes.lower() == 'y':
                        c3_in_c2_in_c1_area = 0  # Initialize
                        c3_not_in_c2_but_in_c1_area = 0  # Initialize
                        if image_c3 is not None:
                            c3_in_c2_in_c1_area = calculate_coloc_area(image_c1, image_c3, label_id, mask_c2=True) #passing the correct number of arguments
                            c3_not_in_c2_but_in_c1_area = calculate_coloc_area(image_c1, image_c3, label_id, mask_c2=False) #passing the correct number of arguments
                        row.extend([c3_in_c2_in_c1_area, c3_not_in_c2_but_in_c1_area])
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
        get_sizes = args.get_sizes
        size_method = args.size_method if get_sizes.lower() == 'y' else None

        if len(set(channels)) < len(channels) or len(channels) < 2 or len(channels) > 3:
            raise ValueError("Channel names must be unique and 2 or 3 channels must be provided.")

        file_lists = {channel: sorted(glob.glob(os.path.join(parent_dir, channel, label_pattern))) for channel, label_pattern in zip(channels, label_patterns)}

        csv_rows = coloc_channels(file_lists, channels, get_sizes, size_method)

        # Create the filename using the selected channels
        channel_string = '_'.join(channels)  # Concatenate channel names with underscores
        csv_file = os.path.join(parent_dir, f'{channel_string}_colocalization.csv')  # create the filename with channel names

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['Filename', f"{channels[0]}_ROI_ID"]
            header.extend([f"{channels[1]}_in_{channels[0]}_count"])
            if get_sizes.lower() == 'y':
                header.extend([f"{channels[0]}_size"])
                if size_method == 'sum':
                    header.extend([f"{channels[1]}_in_{channels[0]}_total_size"])
                if size_method == 'median':
                    header.extend([f"{channels[1]}_in_{channels[0]}_median_size"])
            if len(channels) == 3:
                header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_count",
                               f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_count"])
                if get_sizes.lower() == 'y':
                    if size_method == 'sum':
                        header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_total_size",
                                       f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_total_size"])
                    if size_method == 'median':
                        header.extend([f"{channels[2]}_in_{channels[1]}_in_{channels[0]}_median_size",
                                       f"{channels[2]}_in_{channels[0]}_but_not_{channels[1]}_median_size"])

            writer.writerow(header)
            writer.writerows(csv_rows)

        print(f"Colocalization results saved to {csv_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    if GPU_AVAILABLE:
        print("GPU acceleration is available and will be used.")
    else:
        print("GPU acceleration is not available. Using CPU.")
    main()
