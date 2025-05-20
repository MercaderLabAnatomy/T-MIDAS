import os
import argparse
import glob
from tqdm import tqdm
from tifffile import imwrite, TiffFile
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Batch merge channels with simple sorting')
    parser.add_argument('--input', type=str, required=True, help='Path to the parent folder of the channel folders.')
    parser.add_argument('--channels', nargs='+', type=str, required=True, help='Folder names of all color channels. Example: "TRITC DAPI FITC"')
    parser.add_argument('--time_steps', type=int, default=None, help='Number of time steps if time-lapse (leave empty if static)')
    parser.add_argument('--is_3d', action='store_true', help='Images are 3D (Z dimension present)')
    parser.add_argument('--output_format', type=str, default='python', choices=['python', 'fiji'],
                      help='Format dimension order: python (channel last) or fiji (channel interleaved)')
    return parser.parse_args()

def extract_core_filename(filepath, channel):
    """Extract the core filename by removing the channel name"""
    basename = os.path.basename(filepath)
    # Remove the channel part and clean up
    core = basename.replace(f" - {channel}.tif", "").replace(f"_{channel}.tif", "").replace(f"-{channel}.tif", "")
    return core

def natural_sort_key_for_channel(filepath, channel):
    """Generate a key for natural sorting based on core filename"""
    import re
    core = extract_core_filename(filepath, channel)
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split(r'(\d+)', core)]

def infer_dimension_order(shape, time_steps, is_3d):
    """Infer dimension order from shape, ensuring channel comes first after time"""
    is_timelapse = time_steps is not None
    
    if is_timelapse:
        if is_3d:
            return 'TCZYX'  # Time, Channel, Z, Y, X
        else:
            return 'TCYX'   # Time, Channel, Y, X
    else:
        if is_3d:
            return 'CZYX'   # Channel, Z, Y, X
        else:
            return 'CYX'    # Channel, Y, X




def merge_channels(file_lists, channels, time_steps, is_3d, merged_dir, output_format):
    num_channels = len(channels)
    is_timelapse = time_steps is not None
    
    # Check that all channels have the same number of files
    file_counts = [len(file_lists[channel]) for channel in channels]
    if len(set(file_counts)) > 1:
        raise ValueError(f"Channels have different numbers of files: {dict(zip(channels, file_counts))}")
    
    num_files = file_counts[0]
    if num_files == 0:
        raise ValueError("No files found in any channel")
    
    # Check dimensions of corresponding images across all channels
    print("Checking image dimensions across channels...")
    target_shape = None
    shapes_match = True
    
    for i in range(min(3, num_files)):  # Check first 3 files
        shapes_for_this_file = {}
        for channel in channels:
            with TiffFile(file_lists[channel][i]) as tif:
                img = tif.asarray()
                shapes_for_this_file[channel] = img.shape
                
        if target_shape is None:
            target_shape = shapes_for_this_file[channels[0]]
            
        # Check if all channels have the same shape for this file
        file_shapes = list(shapes_for_this_file.values())
        if len(set(file_shapes)) > 1:
            shapes_match = False
            print(f"File {i}: Different shapes detected:")
            for channel, shape in shapes_for_this_file.items():
                print(f"  {channel}: {shape}")
    
    if not shapes_match:
        print(f"\nWarning: Images have different dimensions. Will resize all to match {channels[0]} channel: {target_shape}")
        resize_needed = True
    else:
        print(f"All images have consistent dimensions: {target_shape}")
        resize_needed = False
    
    # Get information about the first image to determine structure
    with TiffFile(file_lists[channels[0]][0]) as tif:
        img = tif.asarray()

    #print(f"Target image shape: {target_shape}")
    print(f"Structure: {'Time-lapse' if is_timelapse else 'Static'}{f' ({time_steps} steps)' if is_timelapse else ''}, {'3D' if is_3d else '2D'}")

    # Validate time dimension if specified
    if is_timelapse and target_shape[0] != time_steps:
        print(f"Warning: Expected {time_steps} time steps but found {target_shape[0]} in first dimension")

    # Determine output shape - adjust dimension order based on output format
    if output_format == 'fiji':
        # For Fiji/ImageJ, use TZCYX order
        if is_timelapse:
            if is_3d:
                # T, Z, C, Y, X
                time_points, z_slices, height, width = target_shape
                merged_shape = (time_points, z_slices, num_channels, height, width)
                output_dim_order = 'TZCYX'
            else:
                # T, C, Y, X
                time_points, height, width = target_shape
                merged_shape = (time_points, num_channels, height, width)
                output_dim_order = 'TCYX'
        else:
            if is_3d:
                # Z, C, Y, X
                z_slices, height, width = target_shape
                merged_shape = (z_slices, num_channels, height, width)
                output_dim_order = 'ZCYX'
            else:
                # C, Y, X
                height, width = target_shape
                merged_shape = (num_channels, height, width)
                output_dim_order = 'CYX'
    else:
        # For Python format, put channels last (TZYXC, TYXC, ZYXC, YXC)
        if is_timelapse:
            if is_3d:
                # T, Z, Y, X, C
                time_points, z_slices, height, width = target_shape
                merged_shape = (time_points, z_slices, height, width, num_channels)
                output_dim_order = 'TZYXC'
            else:
                # T, Y, X, C
                time_points, height, width = target_shape
                merged_shape = (time_points, height, width, num_channels)
                output_dim_order = 'TYXC'
        else:
            if is_3d:
                # Z, Y, X, C
                z_slices, height, width = target_shape
                merged_shape = (z_slices, height, width, num_channels)
                output_dim_order = 'ZYXC'
            else:
                # Y, X, C
                height, width = target_shape
                merged_shape = (height, width, num_channels)
                output_dim_order = 'YXC'

    print(f"Output shape: {merged_shape}, dimension order: {output_dim_order}")

    # Process each file index
    for i in tqdm(range(num_files), desc='Merging files'):
        # For Python format with channels last, we need a different array structure
        if output_format == 'python':
            # For channel-last format, initialize with channels in last position
            merged_img = np.zeros(merged_shape, dtype=img.dtype)
            
            # Load corresponding file from each channel
            for c, channel in enumerate(channels):
                with TiffFile(file_lists[channel][i]) as tif:
                    channel_img = tif.asarray()
                    
                    # Resize if needed
                    if resize_needed and channel_img.shape != target_shape:
                        from skimage.transform import resize
                        print(f"Resizing {channel} image {i} from {channel_img.shape} to {target_shape}")
                        channel_img = resize(channel_img, target_shape, order=1, preserve_range=True, anti_aliasing=True)
                        channel_img = channel_img.astype(img.dtype)
                    
                    # Place channel in last position based on dimension order
                    if is_timelapse:
                        if is_3d:
                            # TZYXC
                            merged_img[:, :, :, :, c] = channel_img
                        else:
                            # TYXC
                            merged_img[:, :, :, c] = channel_img
                    else:
                        if is_3d:
                            # ZYXC
                            merged_img[:, :, :, c] = channel_img
                        else:
                            # YXC
                            merged_img[:, :, c] = channel_img
        else:
            # Original code for Fiji format (channels first or third position)
            merged_img = np.zeros(merged_shape, dtype=img.dtype)
            
            # Load corresponding file from each channel
            for c, channel in enumerate(channels):
                with TiffFile(file_lists[channel][i]) as tif:
                    channel_img = tif.asarray()
                    
                    # Resize if needed
                    if resize_needed and channel_img.shape != target_shape:
                        from skimage.transform import resize
                        print(f"Resizing {channel} image {i} from {channel_img.shape} to {target_shape}")
                        channel_img = resize(channel_img, target_shape, order=1, preserve_range=True, anti_aliasing=True)
                        channel_img = channel_img.astype(img.dtype)
                    
                    # Place each channel in the correct position (TCZYX or similar)
                    if is_timelapse:
                        if is_3d:
                            merged_img[:, c, :, :, :] = channel_img
                        else:
                            merged_img[:, c, :, :] = channel_img
                    else:
                        if is_3d:
                            merged_img[c, :, :, :] = channel_img
                        else:
                            merged_img[c, :, :] = channel_img

        # Generate output filename based on the first channel's filename
        base_filename = os.path.basename(file_lists[channels[0]][i])
        # Remove channel name from filename if present
        clean_filename = base_filename
        for channel in channels:
            clean_filename = clean_filename.replace(channel, "").replace("--", "-").strip("-")
        
        # Ensure we have a reasonable filename
        if not clean_filename or clean_filename == ".tif":
            clean_filename = f"merged_{i:03d}.tif"
        
        output_filename = os.path.join(merged_dir, clean_filename)
        
        # Save with appropriate metadata for the format
        if output_format == 'fiji':
            # For Fiji/ImageJ, set explicit dimension metadata
            if is_timelapse and is_3d:
                # 5D: TZCYX
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'images': merged_shape[0] * merged_shape[1] * num_channels,
                    'channels': num_channels,
                    'slices': merged_shape[1],  # Z dimension
                    'frames': merged_shape[0],  # T dimension
                    'hyperstack': True,
                    'mode': 'composite',
                    'loop': False,
                    'unit': 'pixel'
                }
            elif is_timelapse:
                # 4D: TCYX
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'images': merged_shape[0] * num_channels,
                    'channels': num_channels,
                    'frames': merged_shape[0],  # T dimension
                    'hyperstack': True,
                    'mode': 'composite',
                    'loop': False,
                    'unit': 'pixel'
                }
            elif is_3d:
                # 4D: ZCYX
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'images': merged_shape[0] * num_channels,
                    'channels': num_channels,
                    'slices': merged_shape[0],  # Z dimension
                    'hyperstack': True,
                    'mode': 'composite',
                    'unit': 'pixel'
                }
            else:
                # 3D: CYX
                imagej_metadata = {
                    'ImageJ': '1.53c',
                    'images': num_channels,
                    'channels': num_channels,
                    'hyperstack': num_channels > 1,
                    'mode': 'composite' if num_channels > 1 else 'grayscale',
                    'unit': 'pixel'
                }
            
            # Write with ImageJ metadata
            imwrite(
                output_filename,
                merged_img,
                imagej=True,
                metadata=imagej_metadata,
                compression='zlib'
            )
        else:
            # For Python format
            metadata = {'axes': output_dim_order}
            imwrite(
                output_filename,
                merged_img,
                metadata=metadata,
                compression='zlib'
            )
    
    print(f"Merging completed successfully. {num_files} files saved.")

def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    time_steps = args.time_steps
    is_3d = args.is_3d
    is_timelapse = time_steps is not None
    output_format = args.output_format

    # Get sorted list of files for each channel (natural sort for proper numeric ordering)
    file_lists = {}
    for channel in channels:
        pattern = os.path.join(parent_dir, channel, '*.tif')
        all_files = sorted(glob.glob(pattern), key=lambda x: natural_sort_key_for_channel(x, channel))
        # Filter out label files
        file_lists[channel] = [f for f in all_files if not f.endswith('_labels.tif')]

    if len(set(channels)) < len(channels) or len(channels) < 2:
        raise ValueError("Channel names must be unique and at least two channels must be provided.")

    print("Configuration:")
    print(f"  Time-lapse: {time_steps if is_timelapse else 'No'}{f' steps' if is_timelapse else ''}")
    print(f"  3D: {is_3d}")
    print(f"  Expected input structure: {'T' if is_timelapse else ''}{'Z' if is_3d else ''}YX")
    print(f"  Output structure: {'T' if is_timelapse else ''}C{'Z' if is_3d else ''}YX")
    print("\nFiles found per channel:")
    for channel in channels:
        print(f"  {channel}: {len(file_lists[channel])}")
        if file_lists[channel]:
            print(f"    First: {os.path.basename(file_lists[channel][0])}")
            print(f"    Last:  {os.path.basename(file_lists[channel][-1])}")

    # Debug: Check if sorting is consistent across channels
    print("\nVerifying sort order consistency:")
    base_order = [os.path.basename(f) for f in file_lists[channels[0]]]
    for channel in channels[1:]:
        channel_order = [os.path.basename(f) for f in file_lists[channel]]
        # Check first few filenames to see if they match the pattern
        for i in range(min(3, len(base_order))):
            base_core = base_order[i].replace(channels[0], "").replace(" - ", " - ").strip()
            channel_core = channel_order[i].replace(channel, "").replace(" - ", " - ").strip()
            if base_core != channel_core:
                print(f"  WARNING: Sort mismatch at position {i}:")
                print(f"    {channels[0]}: {base_order[i]} -> core: '{base_core}'")
                print(f"    {channel}: {channel_order[i]} -> core: '{channel_core}'")
                break
    else:
        print("  Sort order appears consistent across channels")

    # Create output directory
    merged_dir = os.path.join(parent_dir, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)

    print(f"  Output format: {output_format.upper()} ({output_format == 'fiji' and 'channel interleaved (XYCZT)' or 'channel last (Python standard)'})")
    
    merge_channels(file_lists, channels, time_steps, is_3d, merged_dir, output_format)

    print(f"Merged images saved in {merged_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)