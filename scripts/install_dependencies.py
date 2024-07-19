import os
from conda.cli.python_api import run_command, Commands

env_name = "tmidas-env"

# Create the environment
run_command(Commands.CREATE, "-n", env_name, "python=3.8", "-y")

# Activate the environment
os.environ["CONDA_DEFAULT_ENV"] = env_name
os.environ["CONDA_PREFIX"] = os.path.join(os.environ["CONDA_PREFIX"], "envs", env_name)

# Initialize mamba
run_command(Commands.RUN, "-n", env_name, "mamba", "init")

# Install dependencies
dependencies = [
    'numpy', 'scikit-image', 'tifffile', 'pyclesperanto-prototype', 'Pillow',
    'napari-segment-blobs-and-things-with-membranes', 'napari-simpleitk-image-processing',
    'pandas', 'apoc', 'aicsimageio', 'opencv-python', 'readlif', 'SimpleITK',
    'openslide-python', 'glob2', 'pytest', 'cucim', 'aicspylibczi', 'torch',
    'torchvision', 'timm'
]

run_command(Commands.RUN, "-n", env_name, "python", "-m", "pip", "install", "-U", "setuptools", "pip")

run_command(Commands.INSTALL, "-n", env_name, "openslide", "ocl-icd-system", "pyopencl", "cupy", "-y")

run_command(Commands.RUN, "-n", env_name, "pip", "install", "git+https://github.com/ChaoningZhang/MobileSAM.git")

for dependency in dependencies:
    run_command(Commands.RUN, "-n", env_name, "pip", "install", dependency)

run_command(Commands.RUN, "-n", env_name, "python", "-m", "pip", "install", "napari[all]")
run_command(Commands.RUN, "-n", env_name, "python", "-m", "pip", "install", "cellpose")

print("All dependencies installed successfully.")
