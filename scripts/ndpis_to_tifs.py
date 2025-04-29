import openslide
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

"""
Description: This script reads NDPI files and saves them as TIF files.
The script uses the openslide library to read the NDPI files.
The output TIF files are saved in a folder named "tif_files" in the same directory as the input NDPI files.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Convert NDPI files to TIF files.')
    parser.add_argument('--input', type=str, required=True, help='Path to the folder containing the NDPI(s) files.')
    parser.add_argument('--level', type=int, required=True, help='Resolution level of the NDPI image (0 = highest, 1 = second highest, etc).')
    return parser.parse_args()

args = parse_args()
input_folder = args.input
LEVEL = args.level

output_dir = os.path.join(input_folder, "tif_files")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_ndpi_filenames(ndpis_file):
    ndpi_files = []
    with open(ndpis_file, 'r') as f:
        for line in f:
            if line.strip().endswith('.ndpi'):
                # If line is in format "key=filename.ndpi"
                if '=' in line:
                    line = line.split("=", 1)[1]
                ndpi_files.append(line.strip())
    return ndpi_files

def ndpi_2_tif(ndpi_file):
    ndpi_image = openslide.open_slide(os.path.join(input_folder, ndpi_file))
    tiff_image = ndpi_image.read_region((0, 0), LEVEL, ndpi_image.level_dimensions[LEVEL]).convert('L')
    ndpi_image.close()
    return tiff_image

# Find .ndpis files in the input directory
ndpis_files = [f for f in os.listdir(input_folder) if f.endswith(".ndpis")]

if ndpis_files:
    print("Found .ndpis files. Processing NDPI files listed in them...")
    for ndpis_file in tqdm(ndpis_files, desc="Processing .ndpis files"):
        ndpi_files = get_ndpi_filenames(os.path.join(input_folder, ndpis_file))
        for ndpi_file in ndpi_files:
            if ndpi_file.endswith(".ndpi"):
                output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0] + ".tif")
                tiff_image = ndpi_2_tif(ndpi_file)
                tiff_image.save(output_filename)
else:
    print("No .ndpis files found. Processing all .ndpi files in the folder...")
    ndpi_files = [f for f in os.listdir(input_folder) if f.endswith(".ndpi")]
    for ndpi_file in tqdm(ndpi_files, desc="Processing .ndpi files"):
        output_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(ndpi_file))[0] + ".tif")
        tiff_image = ndpi_2_tif(ndpi_file)
        tiff_image.save(output_filename)

print("All done! TIFF files are saved in:", output_dir)
