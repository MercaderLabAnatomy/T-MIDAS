import os
import argparse
import glob
from tifffile import imwrite, TiffFile
import numpy as np
from tqdm import tqdm
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Split TIFF channels with auto time/channel detection')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to folder containing multi-channel TIFFs')
    return parser.parse_args()

def get_t_axis(shape):
    """Find first dimension that looks like time steps (5 <= size < 400)"""
    for i, dim_size in enumerate(shape):
        if 5 <= dim_size < 400:
            return i
    return None  # No axis meets time criteria

def infer_axes(shape, t_axis):
    """Generate axes string following TCZYX priority"""
    ndim = len(shape)
    axes = [''] * ndim
    remaining_dims = list(range(ndim))
    
    # Assign T axis if found
    if t_axis is not None:
        axes[t_axis] = 'T'
        remaining_dims.remove(t_axis)
    
    # Find channel axis (2-4 elements)
    for i in remaining_dims:
        if shape[i] in (2, 3, 4):
            axes[i] = 'C'
            remaining_dims.remove(i)
            break
    
    # Remaining dims: prioritize Z, then Y, then X
    spatial_axes = ['Z', 'Y', 'X']
    for i in remaining_dims:
        if not spatial_axes:
            break
        axes[i] = spatial_axes.pop(0)
    
    # Fill any remaining with generic labels
    for i in remaining_dims:
        if axes[i] == '':
            axes[i] = 'A'  # Anonymous axis
    
    return ''.join(axes)

def split_channels(file_list, output_dir):
    for file_path in tqdm(file_list, desc="Processing"):
        try:
            with TiffFile(file_path) as tif:
                img = tif.asarray()
                metadata = tif.imagej_metadata or {}
                shape = img.shape
                
                # Get existing axes or infer new ones
                axes = metadata.get('axes', '')
                if not axes or len(axes) != len(shape):
                    t_axis = get_t_axis(shape)
                    axes = infer_axes(shape, t_axis)
                
                # Validate axes
                if 'C' not in axes:
                    raise ValueError("No channel axis detected")
                if axes.count('C') > 1:
                    raise ValueError("Multiple channel axes detected")
                
                channel_axis = axes.index('C')
                num_channels = shape[channel_axis]
                
                # Split and save channels
                base = os.path.splitext(os.path.basename(file_path))[0]
                for ch in range(num_channels):
                    channel_img = np.take(img, ch, axis=channel_axis)
                    output_axes = axes.replace('C', '')
                    
                    imwrite(
                        os.path.join(output_dir, f"C{ch}-{base}.tif"),
                        channel_img,
                        imagej=True,
                        metadata={'axes': output_axes},
                        compression='zlib'
                    )
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)

def main():
    args = parse_args()
    input_dir = args.input
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    file_list = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    if not file_list:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")
    
    output_dir = os.path.join(input_dir, 'split_channels')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(file_list)} files with settings:")
    print(" - Auto-detected time axis as first axis that is (5 ≤ dim < 400)")
    print(" - Auto-detected channels (2-4)")
    
    split_channels(file_list, output_dir)
    print(f"\nOutput saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)
