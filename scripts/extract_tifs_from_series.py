import os
import argparse
from tqdm import tqdm
import javabridge
import bioformats
import numpy as np
import tifffile

"""
Description: This script extracts series from a series file and saves them as tif files. 

It should work for all file formats supported by the bioformats library.

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Process a series file.')
    parser.add_argument('--input', type=str, help='path to the folder containing series files')
    return parser.parse_args()

def extract_pixel_resolution(file_path):
    ome_xml_string = bioformats.get_omexml_metadata(path=file_path)
    ome = bioformats.OMEXML(ome_xml_string)
    
    resolutions = {}
    
    for i in range(ome.image_count):
        image = ome.image(i)
        pixels = image.Pixels
        
        resolutions[f'image_{i}'] = {
            'PhysicalSizeX': pixels.PhysicalSizeX,
            'PhysicalSizeXUnit': pixels.PhysicalSizeXUnit,
            'PhysicalSizeY': pixels.PhysicalSizeY,
            'PhysicalSizeYUnit': pixels.PhysicalSizeYUnit,
            'PhysicalSizeZ': pixels.PhysicalSizeZ,
            'PhysicalSizeZUnit': pixels.PhysicalSizeZUnit
        }
    
    return resolutions

def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default

def process_file(file_path, output_directory):
    try:
        pixel_resolutions = extract_pixel_resolution(file_path)
        reader = bioformats.get_image_reader("tmp", path=file_path)
        series_count = reader.rdr.getSeriesCount()

        for series in range(series_count):
            reader.rdr.setSeries(series)
            width = reader.rdr.getSizeX()
            height = reader.rdr.getSizeY()
            z_slices = reader.rdr.getSizeZ()
            channels = reader.rdr.getSizeC()
            timepoints = reader.rdr.getSizeT()
            print(f"Processing series {series} dimensions: X={width}, Y={height}, Z={z_slices}, C={channels}, T={timepoints}")

            output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_series{series}.tif"
            output_path = os.path.join(output_directory, output_filename)

            data = np.zeros((timepoints, z_slices, channels, height, width), dtype=np.uint16)

            for t in tqdm(range(timepoints), desc=f"Reading series {series}"):
                for z in range(z_slices):
                    for c in range(channels):
                        plane = reader.read(c=c, z=z, t=t, series=series, rescale=False)
                        data[t, z, c] = plane
            
            resolution = pixel_resolutions[f'image_{series}']

            # Replace 'µm' with 'um' in the unit fields
            unit = resolution['PhysicalSizeZUnit'] if resolution['PhysicalSizeZUnit'] is not None else 'um'
            unit = unit.replace('µ', 'u') if 'µ' in unit else unit

            resolution_unit = resolution['PhysicalSizeXUnit'] if resolution['PhysicalSizeXUnit'] is not None else 'um'
            resolution_unit = resolution_unit.replace('µ', 'u') if 'µ' in resolution_unit else resolution_unit

            # Metadata including resolution
            metadata = {
                'axes': 'TZCYX',
                'channels': channels,
                'slices': z_slices,
                'frames': timepoints,
                'spacing': safe_float(resolution['PhysicalSizeZ']),
                'unit': unit,
                'pixel_width': safe_float(resolution['PhysicalSizeX']),
                'pixel_height': safe_float(resolution['PhysicalSizeY']),
                'ResolutionUnit': resolution_unit
            }

            # Ensure resolution values are not zero to avoid division by zero
            pixel_width = safe_float(resolution['PhysicalSizeX'])
            pixel_height = safe_float(resolution['PhysicalSizeY'])
            if pixel_width == 0 or pixel_height == 0:
                print(f"Skipping series {series} due to zero resolution values.")
                continue

            tifffile.imwrite(output_path, data, imagej=True, metadata=metadata, 
                             resolution=(1/pixel_width, 1/pixel_height))

            print(f"Series {series} saved to: {output_path}")

        reader.close()
        print(f"Processing complete for {file_path}")

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    input_directory = args.input
    output_directory = os.path.join(input_directory, 'tifs')
    os.makedirs(output_directory, exist_ok=True)

    javabridge.start_vm(class_path=bioformats.JARS)

    try:
        for root, _, files in os.walk(input_directory):
            # Skip the output directory
            if root == output_directory:
                continue

            for file in tqdm(files):
                # exclude tifs from processing
                if file.lower().endswith(('.tif', '.tiff')):
                    continue

                file_path = os.path.join(root, file)
                try:
                    process_file(file_path, output_directory)
                except Exception as e:
                    print(f"An error occurred while processing {file_path}: {str(e)}. Maybe the file type is not supported by bioformats.")
                    import traceback
                    traceback.print_exc()
    finally:
        javabridge.kill_vm()

if __name__ == '__main__':
    main()
