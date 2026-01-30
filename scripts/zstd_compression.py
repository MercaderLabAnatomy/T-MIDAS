import os
import subprocess
import sys
import argparse
from tqdm import tqdm
import shutil

def get_file_size(file_path):
    return os.path.getsize(file_path)

def get_dir_size(dir_path):
    """Calculate total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def compress_file(file_path, remove_source, compression_level):
    compressed_file = f"{file_path}.zst"
    command = ['pzstd', '--quiet']
    
    if compression_level >= 20:
        command.extend(['--ultra', f'-{compression_level}'])
    else:
        command.append(f'-{compression_level}')
    
    if remove_source.lower() == 'y':
        command.append('--rm')
    command.append(file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, compressed_file

def compress_zarr_directory(dir_path, remove_source, compression_level):
    """Compress entire ZARR directory as a single tar.zst archive.
    
    Args:
        dir_path: Path to ZARR directory
        remove_source: Whether to remove source directory after compression
        compression_level: zstd compression level
    """
    tar_file = f"{dir_path}.tar"
    compressed_file = f"{dir_path}.tar.zst"
    
    try:
        # Get original directory size
        original_size = get_dir_size(dir_path)
        
        # Create tar archive
        parent_dir = os.path.dirname(dir_path)
        dir_name = os.path.basename(dir_path)
        
        print(f"Creating tar archive for {dir_name}...")
        tar_command = ['tar', '-cf', tar_file, '-C', parent_dir, dir_name]
        tar_result = subprocess.run(tar_command, capture_output=True, text=True)
        
        if tar_result.returncode != 0:
            print(f"Failed to create tar archive: {tar_result.stderr}", file=sys.stderr)
            return False, None, (original_size, 0)
        
        # Compress the tar file with pzstd
        print(f"Compressing tar archive with zstd (level {compression_level})...")
        command = ['pzstd', '--quiet']
        
        if compression_level >= 20:
            command.extend(['--ultra', f'-{compression_level}'])
        else:
            command.append(f'-{compression_level}')
        
        # Always remove the intermediate tar file
        command.extend(['--rm', tar_file])
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Failed to compress tar archive: {result.stderr}", file=sys.stderr)
            # Clean up tar file if compression failed
            if os.path.exists(tar_file):
                os.remove(tar_file)
            return False, None, (original_size, 0)
        
        # Get compressed size
        compressed_size = get_file_size(compressed_file)
        
        # Remove source directory if requested
        if remove_source.lower() == 'y':
            print(f"Removing source directory {dir_name}...")
            shutil.rmtree(dir_path)
        
        return True, compressed_file, (original_size, compressed_size)
        
    except Exception as e:
        print(f"Error compressing ZARR directory {dir_path}: {str(e)}", file=sys.stderr)
        # Clean up any intermediate files
        if os.path.exists(tar_file):
            os.remove(tar_file)
        return False, None, (0, 0)
    
    # If remove_source and all files compressed successfully, remove empty directories
    if remove_source.lower() == 'y' and fail_count == 0:
        try:
            # Remove empty directories from bottom up
            for root, dirs, files in os.walk(dir_path, topdown=False):
                # Only remove if directory is empty (no uncompressed files)
                try:
                    remaining_files = [f for f in os.listdir(root) if not f.endswith('.zst')]
                    if not remaining_files:
                        for d in dirs:
                            dir_to_remove = os.path.join(root, d)
                            if os.path.exists(dir_to_remove) and not os.listdir(dir_to_remove):
                                os.rmdir(dir_to_remove)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not remove all empty directories: {str(e)}", file=sys.stderr)
    
    return success_count > 0 and fail_count == 0, dir_path, (total_original_size, total_compressed_size)

def find_files(folder, extension):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(f".{extension}"):
                yield os.path.join(root, file)

def find_zarr_directories(folder):
    """Find all ZARR directories (directories ending with .zarr)."""
    zarr_dirs = []
    for root, dirs, _ in os.walk(folder):
        for dir_name in dirs:
            if dir_name.endswith('.zarr'):
                zarr_dirs.append(os.path.join(root, dir_name))
    return zarr_dirs

def main():
    parser = argparse.ArgumentParser(
        description="Compress files using pzstd. Compressed files are saved in the same location as originals with .zst extension.",
        epilog="Example: For file 'data.txt', compressed file will be 'data.txt.zst' in the same directory.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the files to compress")
    parser.add_argument("--file_extension", type=str, required=True, help="File extension to compress (use 'zarr' for ZARR directories)")
    parser.add_argument("--remove_source", type=str, choices=['y', 'n'], default='n', help="Remove the source file after compression? (y/n)")
    parser.add_argument("--compression_level", type=int, choices=range(1, 23), default=3, help="Compression level (1-22)")
    args = parser.parse_args()

    input_folder = args.input_folder
    file_extension = args.file_extension.lstrip('.')
    remove_source = args.remove_source
    compression_level = args.compression_level

    if not os.path.isdir(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist or is not accessible.", file=sys.stderr)
        sys.exit(1)

    # Check if we're looking for ZARR directories
    if file_extension == 'zarr':
        zarr_dirs = find_zarr_directories(input_folder)
        items = zarr_dirs
        is_zarr = True
        print(f"Found {len(items)} ZARR directories in {input_folder} and its subfolders.")
        print(f"Each ZARR directory will be compressed as a single .tar.zst file (e.g., 'data.zarr' -> 'data.zarr.tar.zst')")
    else:
        files = list(find_files(input_folder, file_extension))
        items = files
        is_zarr = False
        print(f"Found {len(items)} files with extension .{file_extension} in {input_folder} and its subfolders.")
        print(f"Compressed files will be saved in the same location with .zst extension (e.g., 'file.{file_extension}' -> 'file.{file_extension}.zst')")

    item_count = len(items)

    if item_count > 0:
        skipped_files = 0
        processed_files = 0
        total_space_freed = 0
        already_compressed = 0
        compression_failed = 0

        desc = "Compressing ZARR directories" if is_zarr else "Compressing files"
        unit = "dir" if is_zarr else "file"

        with tqdm(total=item_count, desc=desc, unit=unit) as pbar:
            for item_path in items:
                if is_zarr:
                    # Check if ZARR directory has already been compressed
                    compressed_file = f"{item_path}.tar.zst"
                    
                    if os.path.exists(compressed_file):
                        skipped_files += 1
                        already_compressed += 1
                        pbar.update(1)
                        continue
                    
                    success, compressed_path, sizes = compress_zarr_directory(item_path, remove_source, compression_level)
                    if success:
                        item_size, compressed_size = sizes
                        space_freed = (item_size - compressed_size) // (1000 * 1000)  # Using SI units (1 MB = 1,000,000 bytes)
                        total_space_freed += space_freed
                        processed_files += 1
                        pbar.set_postfix({"Processed": processed_files, "Skipped": skipped_files, "Space freed": f"{total_space_freed} MB"})
                    else:
                        skipped_files += 1
                        compression_failed += 1
                else:
                    compressed_file = f"{item_path}.zst"
                    
                    if os.path.exists(compressed_file):
                        skipped_files += 1
                        already_compressed += 1
                        pbar.update(1)
                        continue

                    item_size = get_file_size(item_path)
                    success, compressed_file_path = compress_file(item_path, remove_source, compression_level)

                    if success:
                        compressed_size = get_file_size(compressed_file_path)
                        space_freed = (item_size - compressed_size) // (1000 * 1000)  # Using SI units (1 MB = 1,000,000 bytes)
                        total_space_freed += space_freed
                        processed_files += 1
                        pbar.set_postfix({"Processed": processed_files, "Skipped": skipped_files, "Space freed": f"{total_space_freed} MB"})
                    else:
                        print(f"Failed to compress: {item_path}", file=sys.stderr)
                        skipped_files += 1
                        compression_failed += 1

                pbar.update(1)

        print("\nCompression completed.")
        print(f"Total processed files: {processed_files}")
        print(f"Total skipped files: {skipped_files}")
        print(f"  - Already compressed: {already_compressed}")
        print(f"  - Compression failed: {compression_failed}")
        print(f"Total space freed: {total_space_freed} MB")
        print(f"\nCompressed files location: Same directory as original files")
        if is_zarr:
            print(f"ZARR archives saved as: [name].zarr.tar.zst in {input_folder}")
        else:
            print(f"Files saved as: [name].{file_extension}.zst in {input_folder}")
    else:
        if is_zarr:
            print(f"No ZARR directories to compress.")
        else:
            print(f"No files with extension .{file_extension} to compress.")

if __name__ == "__main__":
    main()
