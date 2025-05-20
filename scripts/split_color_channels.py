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
    parser.add_argument('--output_format', type=str, default='python', choices=['python', 'fiji'],
                       help='Format dimension order: python (channel last) or fiji (channel interleaved)')
    return parser.parse_args()


def find_channel_axis(shape, time_steps, is_3d):
    """Find channel axis based on typical microscopy data organization.
    
    In microscopy data, channels are typically:
    - In 5D data: the third dimension (T,Z,C,Y,X)
    - In 4D data without time: the first dimension (C,Z,Y,X)
    - In 4D data without Z: the second dimension (T,C,Y,X)
    - In 3D data: the first dimension (C,Y,X)
    
    Returns the axis index that contains the channel dimension.
    """
    ndim = len(shape)
    is_timelapse = time_steps is not None
    
    # 5D case (T,Z,C,Y,X) or similar
    if ndim == 5:
        # Check if the third dimension (index 2) has a small size (typical for channels)
        if shape[2] <= 16:  # Most microscopy has fewer than 16 channels
            return 2
        # If third dimension is large, check the last dimension
        elif shape[-1] <= 16:
            return ndim - 1
    
    # 4D case
    elif ndim == 4:
        if is_timelapse and is_3d:
            # If both time and Z present, channels might not be represented as a dimension
            # Try the last dimension which could be RGB interleaved
            if shape[-1] <= 16:
                return ndim - 1
            # Otherwise, unsure - return None
            return None
        elif is_timelapse:
            # For (T,C,Y,X) - channel is second dimension
            if shape[1] <= 16:
                return 1
        elif is_3d:
            # For (C,Z,Y,X) - channel is first dimension
            if shape[0] <= 16:
                return 0
            # For (Z,C,Y,X) - channel is second dimension
            elif shape[1] <= 16:
                return 1
    
    # 3D case
    elif ndim == 3:
        # Check first dimension for (C,Y,X)
        if shape[0] <= 16:
            return 0
        # Check last dimension for (Y,X,C) style RGB data
        elif shape[-1] <= 16:
            return ndim - 1
    
    # Last resort: check all dimensions for a small value (3-16) that could be channels
    for i, dim_size in enumerate(shape):
        # Skip dimensions that are likely spatial (Y,X) - typically the last two
        if i >= ndim - 2:
            continue
        # A dimension with size 1-16 that isn't time is likely channels
        if 1 <= dim_size <= 16 and (not is_timelapse or i != 0):
            return i
    
    # If we couldn't find a clear candidate, return None
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

def split_channels(file_list, output_dir, time_steps, is_3d, output_format):
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
                    
                    # For Fiji/ImageJ compatibility
                    if output_format == 'fiji':
                        # For ImageJ, we need axes in TZCYXS order
                        # Since we're removing the C dimension, we need TZYXS
                        # First construct dimension order for the channel image
                        output_dims = list(axes)
                        output_dims.remove('C')  # Remove channel dimension
                        current_order = ''.join(output_dims)
                        
                        # Map to position in array
                        dim_indices = {dim: i for i, dim in enumerate(current_order)}
                        
                        # Construct the transpose order to get TZYXS
                        target_order = ''
                        transpose_indices = []
                        
                        # Add T if exists
                        if 'T' in dim_indices:
                            target_order += 'T'
                            transpose_indices.append(dim_indices['T'])
                            
                        # Add Z if exists  
                        if 'Z' in dim_indices:
                            target_order += 'Z'
                            transpose_indices.append(dim_indices['Z'])
                            
                        # Add Y and X (should always exist)
                        target_order += 'YX'
                        transpose_indices.append(dim_indices['Y'])
                        transpose_indices.append(dim_indices['X'])
                        
                        # Only transpose if order is different
                        if current_order != target_order:
                            channel_img = np.transpose(channel_img, transpose_indices)
                            output_axes = target_order
                        else:
                            output_axes = current_order
                    else:
                        # For Python format, just remove the channel dimension
                        output_axes = axes.replace('C', '')
                    
                    imagej_compatible = (output_format == 'fiji')
                    
                    output_filename = os.path.join(output_dir, f"C{ch}-{base}.tif")
                    imwrite(
                        output_filename,
                        channel_img,
                        imagej=imagej_compatible,
                        metadata={'axes': output_axes} if not imagej_compatible else None,
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
    output_format = args.output_format
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
    
    print(f"  Output format: {output_format.upper()} ({output_format == 'fiji' and 'channel interleaved (XYCZT)' or 'channel last (Python standard)'})")
    
    split_channels(file_list, output_dir, time_steps, is_3d, output_format)
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)