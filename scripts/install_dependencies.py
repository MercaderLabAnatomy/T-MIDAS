import subprocess
import sys
import os

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        sys.exit(1)
    return output.decode()

env_name = "tmidas-env"

# Create the environment
run_command(f"conda create -n {env_name} python=3.8 -y")

# Activate the environment
activate_command = f"conda activate {env_name}"
if sys.platform.startswith('win'):
    activate_command = f"call {activate_command}"

# Set up the command prefix to run in the activated environment
cmd_prefix = f"{activate_command} && "

# Initialize mamba
run_command(cmd_prefix + "mamba init")

# Install dependencies
dependencies = [
    'numpy', 'scikit-image', 'tifffile', 'pyclesperanto-prototype', 'Pillow',
    'napari-segment-blobs-and-things-with-membranes', 'napari-simpleitk-image-processing',
    'pandas', 'apoc', 'aicsimageio', 'opencv-python', 'readlif', 'SimpleITK',
    'openslide-python', 'glob2', 'pytest', 'cucim', 'aicspylibczi', 'torch',
    'torchvision', 'timm'
]

run_command(cmd_prefix + "python -m pip install -U setuptools pip")

run_command(cmd_prefix + "conda install openslide ocl-icd-system pyopencl cupy -y")

run_command(cmd_prefix + "pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

for dependency in dependencies:
    run_command(cmd_prefix + f"pip install {dependency}")

run_command(cmd_prefix + "python -m pip install napari[all]")
run_command(cmd_prefix + "python -m pip install cellpose")

print("All dependencies installed successfully.")
