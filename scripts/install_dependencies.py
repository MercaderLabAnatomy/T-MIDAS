import subprocess
import sys
import os
import json
from tqdm import tqdm

"""
Description: This script installs all dependencies required to run the TMIDAS pipelines.
"""

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        sys.exit(1)
    return output.decode()

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
run_command(cmd_prefix + f"{conda_executable} install openslide ocl-icd-system pyopencl cupy -y")

print("Installing MobileSAM...")
run_command(cmd_prefix + "pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

print("Installing pip packages...")
for dependency in tqdm(dependencies, desc="Installing dependencies"):
    run_command(cmd_prefix + f"pip install {dependency}")

print("Installing napari...")
run_command(cmd_prefix + "python -m pip install napari[all]")

print("Installing cellpose...")
run_command(cmd_prefix + "python -m pip install cellpose")

print("All dependencies installed successfully.")
