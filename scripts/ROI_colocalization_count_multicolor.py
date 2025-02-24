import os
import glob
import csv
import argparse
from tqdm import tqdm
import numpy as np

try:
    import cupy as cp
    import cucim.skimage as cusk
    from cucim import CuImage  # Use CuImage for GPU-accelerated image reading
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    from skimage import io, measure
    GPU_AVAILABLE = False

def parse_args():
    # (Your existing parse_args function remains unchanged)
    pass

def load_image(file_path):
    if GPU_AVAILABLE:
        # Use CuImage for GPU-accelerated image reading
        cu_image = CuImage(file_path)
        return cp.asarray(cu_image)

    else:
        return io.imread(file_path)


def safe_mean(arr):
    return cp.mean(arr).get() if GPU_AVAILABLE and len(arr) > 0 else np.mean(arr) if len(arr) > 0 else 0

def safe_std(arr):
    return cp.std(arr).get() if GPU_AVAILABLE and len(arr) > 0 else np.std(arr) if len(arr) > 0 else 0

def safe_sum(arr):
    return cp.sum(arr).get() if GPU_AVAILABLE and len(arr) > 0 else np.sum(arr) if len(arr) > 0 else 0

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

            label_ids = cp.unique(image_c1) if GPU_AVAILABLE else np.unique(image_c1)
            label_ids = label_ids[label_ids != 0]

            for label_id in label_ids:
                ROI_mask = image_c1 == label_id
                c2_in_c1_count = len(cp.unique(image_c2 * ROI_mask)) - 1 if GPU_AVAILABLE else len(np.unique(image_c2 * ROI_mask)) - 1
                c3_in_c1_count = len(cp.unique(image_c3 * ROI_mask)) - 1 if GPU_AVAILABLE and image_c3 is not None else len(np.unique(image_c3 * ROI_mask)) - 1 if image_c3 is not None else 0

                if image_c3 is not None:
                    c3_in_c2_in_c1_count = len(cp.unique(image_c3 * (image_c2 * ROI_mask))) - 1 if GPU_AVAILABLE else len(np.unique(image_c3 * (image_c2 * ROI_mask))) - 1
                    c3_not_in_c2_but_in_c1_count = len(cp.unique(image_c3 * (ROI_mask & ~image_c2))) - 1 if GPU_AVAILABLE else len(np.unique(image_c3 * (ROI_mask & ~image_c2))) - 1

                if get_areas.lower() == 'y':
                    if GPU_AVAILABLE:
                        props = cusk.measure.regionprops(ROI_mask.astype(cp.int32))
                    else:
                        props = measure.regionprops(ROI_mask.astype(np.int32))
                    area = props[0].area if props else 0
                    
                    if GPU_AVAILABLE:
                        c2_in_c1_areas = [prop.area for prop in cusk.measure.regionprops(cusk.measure.label(image_c2 * ROI_mask).astype(cp.int32))]
                    else:
                        c2_in_c1_areas = [prop.area for prop in measure.regionprops(measure.label(image_c2 * ROI_mask).astype(np.int32))]

                    if area_method == 'average':
                        c2_in_c1_area_value = safe_mean(c2_in_c1_areas)
                        c2_in_c1_std_area = safe_std(c2_in_c1_areas)
                    else:  # sum
                        c2_in_c1_area_value = safe_sum(c2_in_c1_areas)
                        c2_in_c1_std_area = None

                    if image_c3 is not None:
                        if GPU_AVAILABLE:
                            c3_in_c1_areas = [prop.area for prop in cusk.measure.regionprops(cusk.measure.label(image_c3 * ROI_mask).astype(cp.int32))]
                            c3_in_c2_in_c1_areas = [prop.area for prop in cusk.measure.regionprops(cusk.measure.label(image_c3 * (image_c2 * ROI_mask)).astype(cp.int32))]
                        else:
                            c3_in_c1_areas = [prop.area for prop in measure.regionprops(measure.label(image_c3 * ROI_mask).astype(np.int32))]
                            c3_in_c2_in_c1_areas = [prop.area for prop in measure.regionprops(measure.label(image_c3 * (image_c2 * ROI_mask)).astype(np.int32))]

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
    # (Your existing main function remains unchanged)
    pass

if __name__ == "__main__":
    if GPU_AVAILABLE:
        print("GPU acceleration is available and will be used.")
    else:
        print("GPU acceleration is not available. Using CPU.")
    main()
