import os
import subprocess
import sys
import argparse
from tqdm import tqdm

def get_file_size(file_path):
    return os.path.getsize(file_path)

def compress_file(file_path, remove_source):
    compressed_file = f"{file_path}.zst"
    command = ['pzstd', '-19', '--quiet']
    if remove_source.lower() == 'y':
        command.append('--rm')
    command.append(file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0, compressed_file

def find_files(folder, extension):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(f".{extension}"):
                yield os.path.join(root, file)

def main():
    parser = argparse.ArgumentParser(description="Compress files using pzstd.")
    parser.add_argument("--input_folder", type=str, help="Path to the folder containing the files to compress")
    parser.add_argument("--file_extension", type=str, help="File extension to compress")
    parser.add_argument("--remove_source", type=str, default='n', help="Remove the source file after compression? (y/n)")
    args = parser.parse_args()

    input_folder = args.input_folder
    file_extension = args.file_extension
    remove_source = args.remove_source

    # Find files with the specified extension in the given folder and its subfolders
    files = list(find_files(input_folder, file_extension))
    file_count = len(files)

    print(f"Found {file_count} files with extension .{file_extension} in {input_folder} and its subfolders.")

    if file_count > 0:
        skipped_files = 0
        processed_files = 0
        total_space_freed = 0
        already_compressed = 0
        compression_failed = 0

        with tqdm(total=file_count, desc="Compressing files", unit="file") as pbar:
            for file_path in files:
                compressed_file = f"{file_path}.zst"

                if os.path.exists(compressed_file):
                    skipped_files += 1
                    already_compressed += 1
                    pbar.update(1)
                    continue

                file_size = get_file_size(file_path)
                success, compressed_file_path = compress_file(file_path, remove_source)

                if success:
                    compressed_size = get_file_size(compressed_file_path)
                    space_freed = (file_size - compressed_size) // (1024 * 1024)
                    total_space_freed += space_freed
                    processed_files += 1
                    pbar.set_postfix({"Processed": processed_files, "Skipped": skipped_files, "Space Freed": f"{total_space_freed} MB"})
                else:
                    print(f"Failed to compress: {file_path}", file=sys.stderr)
                    skipped_files += 1
                    compression_failed += 1

                pbar.update(1)

        print("\nCompression completed.")
        print(f"Total processed files: {processed_files}")
        print(f"Total skipped files: {skipped_files}")
        print(f"  - Already compressed: {already_compressed}")
        print(f"  - Compression failed: {compression_failed}")
        print(f"Total space freed: {total_space_freed} MB")
    else:
        print(f"No files with extension .{file_extension} to compress.")

if __name__ == "__main__":
    main()
