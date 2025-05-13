import subprocess
import sys
import os
import json
import argparse
from tqdm import tqdm
import multiprocessing

"""
Description: This script installs all dependencies required to run the TMIDAS pipelines.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Install T-MIDAS dependencies")
    parser.add_argument("--skip-jdk-check", action="store_true", help="Skip JDK installation check")
    parser.add_argument("--parallel", action="store_true", help="Install pip packages in parallel where possible")
    parser.add_argument("--max-workers", type=int, default=min(4, multiprocessing.cpu_count()),
                       help="Maximum number of parallel workers (default: min(4, cpu_count))")
    return parser.parse_args()

def run_command(command, show_output=False):
    """Run a shell command and handle output appropriately"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE if not show_output else None,
        stderr=subprocess.PIPE if not show_output else None,
        shell=True,
        text=True
    )
    
    if show_output:
        # Just wait for completion if showing output directly
        return_code = process.wait()
        if return_code != 0:
            print(f"Error executing command: {command}")
            sys.exit(return_code)
        return ""
    else:
        # Capture output if not showing directly
        output, error = process.communicate()
        if process.returncode != 0:
            print(f"Error executing command: {command}")
            print(error)
            sys.exit(process.returncode)
        return output

# Function to check if JDK is installed
def check_jdk_installed():
    try:
        output = run_command("javac -version 2>&1")
        print("JDK is already installed.")
        return True
    except SystemExit:
        print("JDK is not installed.")
        return False

# Function to prompt user to install JDK
def prompt_install_jdk():
    print("To install the required dependencies, please install the default JDK (bioformats needs Java).")
    print("You can do this by running:")
    print("sudo apt install default-jdk")
    
    # Ask for user confirmation to install JDK
    response = input("Do you want to install the default JDK now? (y/n): ").strip().lower()
    if response == 'y':
        run_command("sudo apt update", show_output=True)
        run_command("sudo apt install default-jdk -y", show_output=True)
        
        # Set JAVA_HOME environment variable after installation
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'
        print(f"JAVA_HOME is set to: {os.environ['JAVA_HOME']}")
    else:
        print("JDK installation skipped. The script may not work correctly without JDK.")
        sys.exit(1)

def install_parallel(dependencies, cmd_prefix, max_workers=4):
    """Install pip packages in parallel using multiple processes"""
    print(f"Installing {len(dependencies)} packages using up to {max_workers} workers...")
    
    # Split dependencies into chunks
    chunk_size = max(1, len(dependencies) // max_workers)
    chunks = [dependencies[i:i + chunk_size] for i in range(0, len(dependencies), chunk_size)]
    
    processes = []
    for i, chunk in enumerate(chunks):
        # Combine packages into one pip install command to reduce overhead
        packages = " ".join(chunk)
        cmd = f"{cmd_prefix}python -m pip install {packages}"
        
        print(f"Starting worker {i+1} to install: {packages}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append((process, i+1, chunk))
    
    # Wait for all processes to complete
    with tqdm(total=len(dependencies), desc="Installing packages") as pbar:
        completed = 0
        while processes:
            for i, (process, worker_id, chunk) in enumerate(list(processes)):
                if process.poll() is not None:
                    # Process completed
                    stdout, stderr = process.communicate()
                    if process.returncode != 0:
                        print(f"Error in worker {worker_id}:")
                        print(stderr)
                        sys.exit(1)
                    
                    completed += len(chunk)
                    pbar.update(len(chunk))
                    print(f"Worker {worker_id} completed installing {len(chunk)} packages.")
                    processes.pop(i)
                    break
    
    print("All packages installed successfully!")

def main():
    args = parse_args()
    
    # Check for JDK installation unless skipped
    if not args.skip_jdk_check and not check_jdk_installed():
        prompt_install_jdk()

    # Get the path to the conda executable
    conda_executable = os.path.join(os.path.dirname(sys.executable), 'conda')
    mamba_executable = os.path.join(os.path.dirname(sys.executable), 'mamba')

    env_name = "tmidas-env"

    # Initialize conda
    print("Initializing conda...")
    run_command(f"{conda_executable} init bash")

    # Create the environment
    print(f"Creating environment {env_name}...")
    run_command(f"{conda_executable} create -n {env_name} python=3.8 -y")

    # Get the path to the created environment
    env_path = run_command(f"{conda_executable} env list --json").strip()
    env_path = json.loads(env_path)['envs']
    env_path = [path for path in env_path if path.endswith(env_name)][0]

    # Set up the command prefix to run in the activated environment
    cmd_prefix = f"{conda_executable} run -n {env_name} "

    # Initialize mamba
    print("Initializing mamba...")
    run_command(cmd_prefix + f"{mamba_executable} init")

    # Install dependencies
    dependencies = [
        'numpy', 'scikit-image', 'tifffile', 'pyclesperanto-prototype', 'Pillow',
        'napari-segment-blobs-and-things-with-membranes', 'napari-simpleitk-image-processing',
        'pandas', 'apoc', 'aicsimageio', 'opencv-python', 'readlif', 'SimpleITK',
        'openslide-python', 'glob2', 'pytest', 'cucim', 'aicspylibczi', 'torch',
        'torchvision', 'timm', 'python-javabridge', 'python-bioformats'
    ]

    print("Upgrading pip and setuptools...")
    run_command(cmd_prefix + "python -m pip install -U setuptools pip")

    print("Installing conda packages...")
    run_command(cmd_prefix + f"{conda_executable} install openslide ocl-icd-system pyopencl cupy -y", show_output=True)

    print("Installing MobileSAM...")
    run_command(cmd_prefix + "pip install git+https://github.com/ChaoningZhang/MobileSAM.git", show_output=True)

    print("Installing pip packages...")
    if args.parallel:
        install_parallel(dependencies, cmd_prefix, args.max_workers)
    else:
        for dependency in tqdm(dependencies, desc="Installing dependencies"):
            run_command(cmd_prefix + f"pip install {dependency}")

    print("Installing napari...")
    run_command(cmd_prefix + "python -m pip install napari[all]")

    print("Installing cellpose...")
    run_command(cmd_prefix + "python -m pip install cellpose")

    print("All dependencies installed successfully.")
    print("\nYou can now run T-MIDAS by typing: python ./scripts/user_welcome.py")

if __name__ == "__main__":
    main()