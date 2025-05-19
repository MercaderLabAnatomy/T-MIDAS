import os
import argparse
import glob
from tifffile import imwrite, TiffFile
import numpy as np
from tqdm import tqdm
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Split TIFF channels based on user-specified structure')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to folder containing multi-channel TIFFs')
    parser.add_argument('--time_steps', type=int, default=None,
                       help='Number of time steps if time-lapse (leave empty if static)')
    parser.add_argument('--is_3d', action='store_true',
                       help='Images are 3D (Z dimension present)')
    return parser.parse_args()

def find_channel_axis(shape, time_steps, is_3d):
    """Find channel axis based on user-specified structure and size <= 4"""
    is_timelapse = time_steps is not None
    
    # Expected dimension order: (T)CZ(Y)X or (T)(Z)YXC
    # Channel should be first or last dimension with size <= 4
    
    expected_dims = (is_timelapse and 1 or 0) + (is_3d and 1 or 0) + 2  # +2 for Y,X
    
    if len(shape) < expected_dims:
        raise ValueError(f"Image has {len(shape)} dimensions but expected at least {expected_dims} "
                        f"(T={is_timelapse}, Z={is_3d}, Y, X)")
    
    # Validate time dimension if specified
    if is_timelapse and shape[0] != time_steps:
        print(f"Warning: Expected {time_steps} time steps but found {shape[0]} in first dimension")
    
    # Check if we have more dimensions than expected (likely includes channel)
    if len(shape) > expected_dims:
        # Check first dimension (channel-first: C at position 0 or 1 if timelapse)
        expected_channel_pos = 1 if is_timelapse else 0
        if expected_channel_pos < len(shape) and shape[expected_channel_pos] <= 4:
            return expected_channel_pos
        
        # Check last dimension (channel-last)
        if shape[-1] <= 4:
            return len(shape) - 1
        
        # Check all dimensions for any with size <= 4
        for i, dim_size in enumerate(shape):
            if dim_size <= 4:
                return i
    
    return None

def infer_axes(shape, channel_axis, time_steps, is_3d):
    """Generate axes string based on user input and detected channel position"""
    is_timelapse = time_steps is not None
    ndim = len(shape)
    axes = [''] * ndim
    remaining_dims = list(range(ndim))
    
    # Assign channel axis
    if channel_axis is not None:
        axes[channel_axis] = 'C'
        remaining_dims.remove(channel_axis)
    
    # Assign time axis (always first if present)
    if is_timelapse:
        if 0 not in [channel_axis]:  # If channel is not at position 0
            axes[0] = 'T'
            remaining_dims.remove(0)
        else:
            # If channel is at 0, time should be at 1
            if 1 < len(axes):
                axes[1] = 'T'
                remaining_dims.remove(1)
    
    # Assign spatial axes in order: Z (if 3D), Y, X
    spatial_axes = []
    if is_3d:
        spatial_axes.append('Z')
    spatial_axes.extend(['Y', 'X'])
    
    # Assign remaining dimensions to spatial axes
    for i in remaining_dims:
        if spatial_axes:
            axes[i] = spatial_axes.pop(0)
        else:
            axes[i] = 'A'  # Anonymous if we run out of expected axes
    
    return ''.join(axes)

def split_channels(file_list, output_dir, time_steps, is_3d):
    is_timelapse = time_steps is not None
    
    for file_path in tqdm(file_list, desc="Processing"):
        try:
            with TiffFile(file_path) as tif:
                img = tif.asarray()
                metadata = tif.imagej_metadata or {}
                shape = img.shape
                
                print(f"\nProcessing: {os.path.basename(file_path)}")
                print(f"  Shape: {shape}")
                print(f"  Structure: {'Time-lapse' if is_timelapse else 'Static'}{f' ({time_steps} steps)' if is_timelapse else ''}, {'3D' if is_3d else '2D'}")
                
                # Find channel axis
                channel_axis = find_channel_axis(shape, time_steps, is_3d)
                
                if channel_axis is None:
                    raise ValueError(f"No channel axis detected in shape {shape}. "
                                   f"Expected channel dimension ≤ 4")
                
                # Generate axes string
                axes = infer_axes(shape, channel_axis, time_steps, is_3d)
                
                num_channels = shape[channel_axis]
                print(f"  Detected axes: {axes}")
                print(f"  Channel axis: {channel_axis} (size: {num_channels})")
                
                # Validate that we found channel axis
                if 'C' not in axes:
                    raise ValueError("Failed to assign channel axis")
                
                # Split and save channels
                base = os.path.splitext(os.path.basename(file_path))[0]
                for ch in range(num_channels):
                    channel_img = np.take(img, ch, axis=channel_axis)
                    output_axes = axes.replace('C', '')
                    
                    output_filename = os.path.join(output_dir, f"C{ch}-{base}.tif")
                    imwrite(
                        output_filename,
                        channel_img,
                        imagej=True,
                        metadata={'axes': output_axes},
                        compression='zlib'
                    )
                    
                print(f"  Saved {num_channels} channels")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)

def main():
    args = parse_args()
    input_dir = args.input
    time_steps = args.time_steps
    is_3d = args.is_3d
    is_timelapse = time_steps is not None
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    file_list = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    if not file_list:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")
    
    output_dir = os.path.join(input_dir, 'split_channels')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(file_list)} files:")
    print(f"  Time-lapse: {time_steps if is_timelapse else 'No'}{f' steps' if is_timelapse else ''}")
    print(f"  3D: {is_3d}")
    print(f"  Expected structure: {'T' if is_timelapse else ''}{'C' if True else ''}{'Z' if is_3d else ''}YX")
    
    split_channels(file_list, output_dir, time_steps, is_3d)
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)