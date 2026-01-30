import os
import subprocess
import sys
import argparse
from tqdm import tqdm

"""
This script decompresses files using pzstd.
For ZARR directories compressed as .tar.zst, it will decompress and extract them.
"""



def get_file_size(file_path):
    return os.path.getsize(file_path)

def decompress_file(file_path, remove_compressed):
    command = ['pzstd', '--quiet', '--decompress']
    
    if remove_compressed.lower() == 'y':
        command.append('--rm')
    
    command.append(file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0

def decompress_zarr_archive(archive_path, remove_compressed):
    """Decompress a ZARR .tar.zst archive and extract it.
    
    Args:
        archive_path: Path to .tar.zst archive
        remove_compressed: Whether to remove compressed archive after decompression
    """
    tar_file = archive_path.rstrip('.zst')  # Remove .zst to get .tar
    
    try:
        # Decompress the .zst file
        print(f"Decompressing {os.path.basename(archive_path)}...")
        command = ['pzstd', '--quiet', '--decompress']
        
        # Always keep the .zst file initially, we'll handle removal later
        command.append(archive_path)
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Failed to decompress: {result.stderr}", file=sys.stderr)
            return False
        
        # Extract the tar archive
        print(f"Extracting tar archive...")
        parent_dir = os.path.dirname(tar_file)
        tar_command = ['tar', '-xf', tar_file, '-C', parent_dir]
        tar_result = subprocess.run(tar_command, capture_output=True, text=True)
        
        if tar_result.returncode != 0:
            print(f"Failed to extract tar archive: {tar_result.stderr}", file=sys.stderr)
            # Clean up tar file
            if os.path.exists(tar_file):
                os.remove(tar_file)
            return False
        
        # Clean up the intermediate tar file
        if os.path.exists(tar_file):
            os.remove(tar_file)
        
        # Remove the compressed archive if requested
        if remove_compressed.lower() == 'y' and os.path.exists(archive_path):
            os.remove(archive_path)
        
        return True
        
    except Exception as e:
        print(f"Error decompressing ZARR archive {archive_path}: {str(e)}", file=sys.stderr)
        # Clean up any intermediate files
        if os.path.exists(tar_file):
            os.remove(tar_file)
        return False

def find_files(folder, extension):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(f".{extension}"):
                yield os.path.join(root, file)

def find_zarr_archives(folder):
    """Find all ZARR .tar.zst archive files."""
    zarr_archives = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.zarr.tar.zst'):
                file_path = os.path.join(root, file)
                zarr_archives.append(file_path)
    return zarr_archives

def main():
    parser = argparse.ArgumentParser(
        description="Decompress files using pzstd. Decompressed files are saved in the same location, removing .zst extension.",
        epilog="Example: For file 'data.txt.zst', decompressed file will be 'data.txt' in the same directory.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the files to decompress")
    parser.add_argument("--file_extension", type=str, required=True, help="File extension to decompress (use 'zarr' for ZARR directories, 'zst' for regular files)")
    parser.add_argument("--remove_compressed", type=str, choices=['y', 'n'], default='n', help="Remove the compressed file after decompression? (y/n)")
    args = parser.parse_args()

    input_folder = args.input_folder
    file_extension = args.file_extension.lstrip('.')
    remove_compressed = args.remove_compressed

    if not os.path.isdir(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist or is not accessible.", file=sys.stderr)
        sys.exit(1)

    # Check if we're looking for compressed ZARR archives
    if file_extension == 'zarr':
        zarr_archives = find_zarr_archives(input_folder)
        items = zarr_archives
        is_zarr = True
        print(f"Found {len(items)} compressed ZARR archives in {input_folder} and its subfolders.")
        print(f"ZARR directories will be extracted from .tar.zst archives (e.g., 'data.zarr.tar.zst' -> 'data.zarr/')")
    else:
        files = list(find_files(input_folder, file_extension))
        items = files
        is_zarr = False
        print(f"Found {len(items)} files with extension .{file_extension} in {input_folder} and its subfolders.")
        print(f"Decompressed files will be saved in the same location, removing .zst extension")

    item_count = len(items)

    if item_count > 0:
        processed_files = 0
        skipped_files = 0
        decompression_failed = 0

        desc = "Decompressing ZARR archives" if is_zarr else "Decompressing files"
        unit = "archive" if is_zarr else "file"

        with tqdm(total=item_count, desc=desc, unit=unit) as pbar:
            for item_path in items:
                if is_zarr:
                    success = decompress_zarr_archive(item_path, remove_compressed)
                else:
                    success = decompress_file(item_path, remove_compressed)

                if success:
                    processed_files += 1
                else:
                    print(f"Failed to decompress: {item_path}", file=sys.stderr)
                    skipped_files += 1
                    decompression_failed += 1

                pbar.update(1)

        print("\nDecompression completed.")
        print(f"Total processed files: {processed_files}")
        print(f"Total skipped files: {skipped_files}")
        print(f"  - Decompression failed: {decompression_failed}")
        print(f"\nDecompressed files location: Same directory as compressed files")
        if is_zarr:
            print(f"ZARR directories extracted to: {input_folder}")
        else:
            print(f"Files saved as: [name].{file_extension.rstrip('.zst')} in {input_folder}")
    else:
        if is_zarr:
            print(f"No compressed ZARR archives (.zarr.tar.zst) to decompress.")
        else:
            print(f"No files with extension .{file_extension} to decompress.")

if __name__ == "__main__":
    main()
