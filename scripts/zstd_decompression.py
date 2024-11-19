import os
import subprocess
import sys
import argparse
from tqdm import tqdm

def get_file_size(file_path):
    return os.path.getsize(file_path)

def decompress_file(file_path, remove_compressed):
    command = ['pzstd', '--quiet', '--decompress']
    
    if remove_compressed.lower() == 'y':
        command.append('--rm')
    
    command.append(file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0

def find_files(folder, extension):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(f".{extension}"):
                yield os.path.join(root, file)

def main():
    parser = argparse.ArgumentParser(description="Decompress files using unzstd.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the files to decompress")
    parser.add_argument("--file_extension", type=str, required=True, help="File extension to decompress (default: zst)")
    parser.add_argument("--remove_compressed", type=str, choices=['y', 'n'], default='n', help="Remove the compressed file after decompression? (y/n)")
    args = parser.parse_args()

    input_folder = args.input_folder
    file_extension = args.file_extension.lstrip('.')
    remove_compressed = args.remove_compressed

    if not os.path.isdir(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist or is not accessible.", file=sys.stderr)
        sys.exit(1)

    files = list(find_files(input_folder, file_extension))
    file_count = len(files)

    print(f"Found {file_count} files with extension .{file_extension} in {input_folder} and its subfolders.")

    if file_count > 0:
        processed_files = 0
        skipped_files = 0
        decompression_failed = 0

        with tqdm(total=file_count, desc="Decompressing files", unit="file") as pbar:
            for file_path in files:
                success = decompress_file(file_path, remove_compressed)

                if success:
                    processed_files += 1
                else:
                    print(f"Failed to decompress: {file_path}", file=sys.stderr)
                    skipped_files += 1
                    decompression_failed += 1

                pbar.update(1)

        print("\nDecompression completed.")
        print(f"Total processed files: {processed_files}")
        print(f"Total skipped files: {skipped_files}")
        print(f"  - Decompression failed: {decompression_failed}")
    else:
        print(f"No files with extension .{file_extension} to decompress.")

if __name__ == "__main__":
    main()
