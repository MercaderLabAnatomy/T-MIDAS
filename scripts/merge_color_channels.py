import os
import argparse
import glob
from tifffile import imwrite, TiffFile
import numpy as np
from tqdm import tqdm
import sys
import re
from collections import Counter
import difflib

def parse_args():
    parser = argparse.ArgumentParser(description='Merge single-channel TIFFs into multi-channel images')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the parent folder containing the channel folders')
    parser.add_argument('--channels', nargs='+', type=str, required=True, 
                        help='Names of the channel folders to merge')
    parser.add_argument('--time_steps', type=int, default=None, 
                        help='Number of time steps if time-lapse (leave empty if not a timelapse)')
    parser.add_argument('--is_3d', action='store_true', 
                        help='Set this flag if images are 3D (with Z dimension)')
    parser.add_argument('--output_format', type=str, default='python', choices=['python', 'fiji'],
                        help='Output dimension ordering: python (channel last) or fiji (channel interleaved)')
    return parser.parse_args()

def find_common_patterns(filenames):
    """
    Find common prefixes and suffixes in a list of filenames
    
    Args:
        filenames: List of filenames to analyze
        
    Returns:
        tuple: (prefix, suffix) - The common prefix and suffix found
    """
    if not filenames:
        return "", ""
    
    # Find common prefix
    prefix = os.path.commonprefix(filenames)
    
    # Find common suffix (check from the end)
    reversed_names = [name[::-1] for name in filenames]
    suffix = os.path.commonprefix(reversed_names)[::-1]
    
    # Remove extension from suffix calculation if all files have same extension
    extensions = [os.path.splitext(name)[1] for name in filenames]
    if len(set(extensions)) == 1:
        suffix = suffix[:-len(extensions[0])] + extensions[0]
    
    return prefix, suffix

def extract_channel_patterns(channel, file_paths):
    """
    Analyze files in a channel to find common patterns
    
    Args:
        channel: Name of the channel
        file_paths: List of file paths for this channel
        
    Returns:
        tuple: (prefix, suffix) - The common prefix and suffix found
    """
    # Extract just the basenames
    filenames = [os.path.basename(path) for path in file_paths]
    
    # Find common patterns
    prefix, suffix = find_common_patterns(filenames)
    
    # If channel name appears in the suffix or prefix, make sure it's included
    # This helps when the channel name itself isn't present in all filenames
    if channel.lower() in suffix.lower() and channel.lower() not in prefix.lower():
        # Find where the channel name appears in the suffix
        pattern = re.compile(re.escape(channel), re.IGNORECASE)
        match = pattern.search(suffix)
        if match:
            # Include enough of the suffix to capture the channel name and surrounding pattern
            start_idx = max(0, match.start() - 5)  # Include up to 5 chars before channel name
            channel_suffix = suffix[start_idx:]
            suffix = channel_suffix
    
    if channel.lower() in prefix.lower() and channel.lower() not in suffix.lower():
        # Find where the channel name appears in the prefix
        pattern = re.compile(re.escape(channel), re.IGNORECASE)
        match = pattern.search(prefix)
        if match:
            # Include enough of the prefix to capture the channel name and surrounding pattern
            end_idx = min(len(prefix), match.end() + 5)  # Include up to 5 chars after channel name
            channel_prefix = prefix[:end_idx]
            prefix = channel_prefix
    
    # Check if we found anything useful
    if not prefix and not suffix:
        print(f"Warning: No common patterns found in {channel} filenames")
        # As fallback, just use the channel name itself
        if any(channel.lower() in f.lower() for f in filenames):
            print(f"Using channel name '{channel}' as pattern")
            return "", channel
    
    return prefix, suffix

def extract_core_filename(filename, prefix="", suffix=""):
    """
    Extract the core part of a filename by removing prefix and suffix
    
    Args:
        filename: The filename to process
        prefix: Prefix to remove
        suffix: Suffix to remove
        
    Returns:
        str: The core part of the filename
    """
    basename = os.path.basename(filename)
    
    # Remove prefix if it exists
    if prefix and basename.startswith(prefix):
        basename = basename[len(prefix):]
    
    # Remove suffix if it exists
    if suffix and basename.endswith(suffix):
        basename = basename[:-len(suffix)]
    
    # Ensure we have an extension
    if not os.path.splitext(basename)[1]:
        basename += ".tif"
        
    return basename

def get_matching_files(channels, parent_dir):
    """
    Find matching files across channels based on their core filenames
    
    Args:
        channels: List of channel names
        parent_dir: Parent directory containing channel folders
        
    Returns:
        list: List of tuples (core, {channel: file_path, ...})
    """
    # Dictionary to hold all files for each channel
    all_files = {}
    
    # Dictionary to store common patterns for each channel
    channel_patterns = {}
    
    # Load files for each channel and find patterns
    for channel in channels:
        channel_dir = os.path.join(parent_dir, channel)
        if not os.path.isdir(channel_dir):
            raise ValueError(f"Channel directory not found: {channel_dir}")
        
        # Get all TIF files in this channel, excluding label files
        all_files[channel] = [
            f for f in sorted(glob.glob(os.path.join(channel_dir, '*.tif')))
            if not f.endswith('_labels.tif')
        ]
        
        if not all_files[channel]:
            raise ValueError(f"No TIF files found in channel directory: {channel_dir}")
        
        # Find channel-specific patterns
        prefix, suffix = extract_channel_patterns(channel, all_files[channel])
        channel_patterns[channel] = (prefix, suffix)
        
        print(f"Channel {channel} patterns:")
        print(f"  Prefix: '{prefix}'")
        print(f"  Suffix: '{suffix}'")
        
        # Show example of core extraction
        if all_files[channel]:
            sample = os.path.basename(all_files[channel][0])
            core = extract_core_filename(sample, prefix, suffix)
            print(f"  Sample: '{sample}' → Core: '{core}'")
    
    # Create a mapping from core filename to actual file for each channel
    core_mapping = {}
    for channel in channels:
        prefix, suffix = channel_patterns[channel]
        core_mapping[channel] = {}
        
        for file_path in all_files[channel]:
            filename = os.path.basename(file_path)
            core = extract_core_filename(filename, prefix, suffix)
            core_mapping[channel][core] = file_path
    
    # Find potential matches across channels
    all_cores = {}
    for channel in channels:
        all_cores[channel] = set(core_mapping[channel].keys())
    
    # First try: Find exact matches across all channels
    common_cores = set.intersection(*all_cores.values()) if all_cores else set()
    
    # If no exact matches, try fuzzy matching
    if not common_cores:
        print("Warning: No exact core matches found across channels. Trying fuzzy matching...")
        
        # Get all cores from first channel as reference
        reference_channel = channels[0]
        reference_cores = list(all_cores[reference_channel])
        
        # For each reference core, find best matches in other channels
        fuzzy_matches = []
        for ref_core in reference_cores:
            match_found = True
            matches = {reference_channel: core_mapping[reference_channel][ref_core]}
            
            for channel in channels[1:]:
                # Find best match in this channel
                best_match = None
                best_score = 0
                
                for core in all_cores[channel]:
                    score = difflib.SequenceMatcher(None, ref_core, core).ratio()
                    if score > best_score and score > 0.6:  # Require 60%+ similarity
                        best_score = score
                        best_match = core
                
                if best_match:
                    matches[channel] = core_mapping[channel][best_match]
                else:
                    match_found = False
                    break
            
            if match_found:
                fuzzy_matches.append((ref_core, matches))
        
        if fuzzy_matches:
            print(f"Found {len(fuzzy_matches)} fuzzy matches")
            return fuzzy_matches
        else:
            raise ValueError("No matching files found across channels, even with fuzzy matching")
    
    # Convert exact matches to sorted list for consistent order
    exact_matches = []
    for core in sorted(common_cores):
        matches = {channel: core_mapping[channel][core] for channel in channels}
        exact_matches.append((core, matches))
    
    print(f"Found {len(exact_matches)} exact matches across all channels")
    return exact_matches

def merge_channels(parent_dir, channels, time_steps, is_3d, output_format):
    """Merge matching files from different channels"""
    is_timelapse = time_steps is not None
    
    # Create output directory
    output_dir = os.path.join(parent_dir, "merged")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get matching files across all channels
    file_groups = get_matching_files(channels, parent_dir)
    
    if not file_groups:
        print("No matching files found across channels.")
        return
    
    # Process each group of matching files
    for core, channel_files in tqdm(file_groups, desc="Merging files"):
        try:
            # Determine output shape and type for THIS group
            sample_channel = channels[0]
            sample_file = channel_files[sample_channel]
            with TiffFile(sample_file) as tif:
                sample_img = tif.asarray()
                sample_shape = sample_img.shape
                sample_dtype = sample_img.dtype

            # Now use sample_shape and sample_dtype for this group
            if output_format == 'python':

                # For python: TZYXC or similar, with channel last
                if is_timelapse:
                    if is_3d:
                        # T,Z,Y,X,C
                        merged_shape = sample_shape + (len(channels),)
                    else:
                        # T,Y,X,C
                        merged_shape = sample_shape + (len(channels),)
                else:
                    if is_3d:
                        # Z,Y,X,C
                        merged_shape = sample_shape + (len(channels),)
                    else:
                        # Y,X,C
                        merged_shape = sample_shape + (len(channels),)
            else:
                # For fiji: TZCYX or similar, with channel in 3rd position
                if is_timelapse:
                    if is_3d:
                        # T,Z,C,Y,X
                        time_points, z_slices = sample_shape[0], sample_shape[1]
                        height, width = sample_shape[2], sample_shape[3]
                        merged_shape = (time_points, z_slices, len(channels), height, width)
                    else:
                        # T,C,Y,X
                        time_points = sample_shape[0]
                        height, width = sample_shape[1], sample_shape[2]
                        merged_shape = (time_points, len(channels), height, width)
                else:
                    if is_3d:
                        # Z,C,Y,X
                        z_slices = sample_shape[0]
                        height, width = sample_shape[1], sample_shape[2]
                        merged_shape = (z_slices, len(channels), height, width)
                    else:
                        # C,Y,X
                        height, width = sample_shape[0], sample_shape[1]
                        merged_shape = (len(channels), height, width)
            
            # Create the merged image array
            merged_img = np.zeros(merged_shape, dtype=sample_dtype)
            
            # Load each channel and add to the merged image
            for c, channel in enumerate(channels):
                file_path = channel_files[channel]
                try:
                    with TiffFile(file_path) as tif:
                        channel_img = tif.asarray()
                    
                    # Check if dimensions match
                    if channel_img.shape != sample_shape:
                        raise ValueError(f"Image dimensions don't match: expected {sample_shape}, got {channel_img.shape}")
                    
                    # Add channel to merged image in the appropriate position
                    if output_format == 'python':
                        # For Python format, channel is last dimension
                        if is_timelapse:
                            if is_3d:
                                # TZYXC
                                merged_img[..., c] = channel_img
                            else:
                                # TYXC
                                merged_img[..., c] = channel_img
                        else:
                            if is_3d:
                                # ZYXC
                                merged_img[..., c] = channel_img
                            else:
                                # YXC
                                merged_img[..., c] = channel_img
                    else:
                        # For Fiji format, channel is third dimension
                        if is_timelapse:
                            if is_3d:
                                # TZCYX
                                merged_img[:, :, c, :, :] = channel_img
                            else:
                                # TCYX
                                merged_img[:, c, :, :] = channel_img
                        else:
                            if is_3d:
                                # ZCYX
                                merged_img[:, c, :, :] = channel_img
                            else:
                                # CYX
                                merged_img[c, :, :] = channel_img
                                
                except Exception as e:
                    print(f"Error loading channel {channel}: {str(e)}")
                    raise
            
            # Construct output filename (use core without any extension)
            core_basename = os.path.splitext(core)[0]
            output_filename = os.path.join(output_dir, f"{core_basename}.tif")
            
            # Save the merged image
            # Simple dimension metadata for Python format
            metadata = {'axes': 'TZYXC' if is_timelapse and is_3d else 
                               'TYXC' if is_timelapse else 
                               'ZYXC' if is_3d else 'YXC'}
            imwrite(output_filename, merged_img, metadata=metadata, compression='zlib')

            if output_format == 'fiji':
                # Set ImageJ metadata
                imagej_metadata = {'ImageJ': '1.53c'}
                
                if is_timelapse and is_3d:
                    # 5D: TZCYX
                    imagej_metadata.update({
                        'images': merged_shape[0] * merged_shape[1] * len(channels),
                        'channels': len(channels),
                        'slices': merged_shape[1],  # Z dimension
                        'frames': merged_shape[0],  # T dimension
                        'hyperstack': True,
                        'mode': 'composite'
                    })
                elif is_timelapse:
                    # 4D: TCYX
                    imagej_metadata.update({
                        'images': merged_shape[0] * len(channels),
                        'channels': len(channels),
                        'frames': merged_shape[0],  # T dimension
                        'hyperstack': True,
                        'mode': 'composite'
                    })
                elif is_3d:
                    # 4D: ZCYX
                    imagej_metadata.update({
                        'images': merged_shape[0] * len(channels),
                        'channels': len(channels),
                        'slices': merged_shape[0],  # Z dimension
                        'hyperstack': True,
                        'mode': 'composite'
                    })
                else:
                    # 3D: CYX
                    imagej_metadata.update({
                        'images': len(channels),
                        'channels': len(channels),
                        'hyperstack': len(channels) > 1,
                        'mode': 'composite' if len(channels) > 1 else 'grayscale'
                    })
                
                imwrite(output_filename, merged_img, imagej=True, metadata=imagej_metadata, compression='zlib')
                
                
        except Exception as e:
            print(f"Error processing {core}: {str(e)}")
            continue

def main():
    args = parse_args()
    parent_dir = args.input
    channels = [c.upper() for c in args.channels]
    time_steps = args.time_steps
    is_3d = args.is_3d
    output_format = args.output_format
    
    # Validate inputs
    if len(set(channels)) < len(channels):
        raise ValueError("Channel names must be unique")
    
    if len(channels) < 2:
        raise ValueError("At least two channels must be provided")
    
    print("Configuration:")
    print(f"  Channels: {', '.join(channels)}")
    print(f"  Time-lapse: {time_steps if time_steps else 'No'}")
    print(f"  3D: {is_3d}")
    print(f"  Output format: {output_format}")
    
    # Print channel file information
    print("\nChecking channel contents:")
    for channel in channels:
        channel_dir = os.path.join(parent_dir, channel)
        if not os.path.isdir(channel_dir):
            print(f"  Warning: Channel directory not found: {channel_dir}")
            continue
            
        files = [f for f in sorted(glob.glob(os.path.join(channel_dir, '*.tif')))
                if not f.endswith('_labels.tif')]
        
        print(f"  {channel}: {len(files)} files")
        if files:
            print(f"    First: {os.path.basename(files[0])}")
            print(f"    Last: {os.path.basename(files[-1])}")
    
    # Merge the channels
    merge_channels(parent_dir, channels, time_steps, is_3d, output_format)
    
    print(f"\nProcessing complete. Merged files saved to {os.path.join(parent_dir, 'merged')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)