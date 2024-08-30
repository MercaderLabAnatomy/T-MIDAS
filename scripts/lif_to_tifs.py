import os
import argparse
from tqdm import tqdm
import javabridge
import bioformats
import numpy as np
import tifffile



def parse_args():
    parser = argparse.ArgumentParser(description='Process a lif file.')
    parser.add_argument('--input', type=str, help='path to the folder containing lif files')
    return parser.parse_args()

def extract_pixel_resolution(lif_file_path):
    ome_xml_string = bioformats.get_omexml_metadata(path=lif_file_path)
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



def process_lif_file(lif_file_path, output_directory):
    try:
        pixel_resolutions = extract_pixel_resolution(lif_file_path)
        reader = bioformats.get_image_reader("tmp", path=lif_file_path)
        series_count = reader.rdr.getSeriesCount()

        for series in range(series_count):
            reader.rdr.setSeries(series)
            width = reader.rdr.getSizeX()
            height = reader.rdr.getSizeY()
            z_slices = reader.rdr.getSizeZ()
            channels = reader.rdr.getSizeC()
            timepoints = reader.rdr.getSizeT()
            print(f"Processing series {series} dimensions: X={width}, Y={height}, Z={z_slices}, C={channels}, T={timepoints}")

            output_filename = f"series_{series}.tif"
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
                'pixel_width': safe_float( resolution['PhysicalSizeX']),
                'pixel_height': safe_float( resolution['PhysicalSizeY']),
                'ResolutionUnit': resolution_unit
            }

            tifffile.imwrite(output_path, data, imagej=True, metadata=metadata, 
                             resolution=(1/safe_float(resolution['PhysicalSizeX']), 1/safe_float(resolution['PhysicalSizeY'])))

            print(f"Series {series} saved to: {output_path}")

        reader.close()
        print(f"Processing complete for {lif_file_path}")

    except Exception as e:
        print(f"An error occurred while processing {lif_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()





def main():
    args = parse_args()
    input_folder = args.input

    lif_files = [file for file in os.listdir(input_folder) if file.endswith(".lif")]

    try:
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        
        for lif_file in tqdm(lif_files, total=len(lif_files), desc="Processing lif files"):
            lif_path = os.path.join(input_folder, lif_file)
            output_dir = os.path.join(input_folder, lif_file.split(".")[0])
            os.makedirs(output_dir, exist_ok=True)
            process_lif_file(lif_path, output_dir)

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")
    finally:
        javabridge.kill_vm()

if __name__ == "__main__":
    main()
