import os
import argparse
from tqdm import tqdm
import javabridge as jb
import bioformats
import numpy as np
import tifffile


"""
Description: This script extracts series from a series file and saves them as tif files. 
It should work for all file formats supported by the bioformats library.
The user has the option to filter series by a string in the series name.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Process a series file.')
    parser.add_argument('--input', type=str, help='path to the folder containing series files')
    parser.add_argument('--filter', type=str, nargs='?', default='', help='string to filter series names (optional)')
    parser.add_argument('--file_format', type=str, required=True, help='file format to process (e.g., .lif, .czi)')
    return parser.parse_args()

def extract_pixel_resolution(file_path: str) -> dict:
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

def extract_series_info(file_path: str, filter_string: str) -> dict:
    ome_xml_string = bioformats.get_omexml_metadata(path=file_path)
    ome = bioformats.OMEXML(ome_xml_string)
    
    series_info = {}
    
    for i in range(ome.image_count):
        image = ome.image(i)
        series_name = image.get_Name()
        
        if not filter_string or filter_string in series_name:
            series_info[i] = series_name
            #print(f"Found series {i} ({series_name})")

    return series_info


def process_file(file_path: str, output_directory: str, filter_string: str):
    try:
        pixel_resolutions = extract_pixel_resolution(file_path)
        series_info = extract_series_info(file_path, filter_string)
        
        if not series_info:
            print(f"No series found matching filter '{filter_string}' in {file_path}.")
            return
        
        reader = bioformats.get_image_reader("tmp", path=file_path)
        
        for series, series_name in series_info.items():
            reader.rdr.setSeries(series)
            
            width = reader.rdr.getSizeX()
            height = reader.rdr.getSizeY()
            z_slices = reader.rdr.getSizeZ()
            channels = reader.rdr.getSizeC()
            timepoints = reader.rdr.getSizeT()
            print(f"Processing series {series} ({series_name}) dimensions: X={width}, Y={height}, Z={z_slices}, C={channels}, T={timepoints}")

            # Use series number instead of series name
            output_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_series_{str(series).zfill(2)}.tif"
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

def init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    https://github.com/pskeshu/microscoper/blob/master/microscoper/io.py#L141-L162
    """
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)



def main():
    args = parse_args()
    input_directory = args.input
    filter_string = args.filter
    file_format = args.file_format.lstrip('.')  
    output_directory = os.path.join(input_directory, 'tifs')
    os.makedirs(output_directory, exist_ok=True)

    jb.start_vm(class_path=bioformats.JARS)
    init_logger()

    try:
        for root, _, files in os.walk(input_directory):
            # Skip the output directory
            if root == output_directory:
                continue

            for file in tqdm(files):
                # Only process files with the specified format
                if not file.lower().endswith(file_format.lower()):
                    continue

                if file.lower().endswith(('.tif', '.tiff')):  # Exclude tif files from processing
                    continue

                file_path = os.path.join(root, file)
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    try:
                        process_file(file_path, output_directory, filter_string)
                    except Exception as e:
                        print(f"An error occurred while processing {file_path}: {str(e)}. Skipping this file.")
                        import traceback
                        traceback.print_exc()
    finally:
        jb.kill_vm()

if __name__ == '__main__':
    main()
