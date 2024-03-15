import subprocess

dependencies = [
    'numpy',
    'scikit-image',
    'tifffile',
    'pyclesperanto-prototype', 
    'Pillow',
    'napari-segment-blobs-and-things-with-membranes',
    'napari-simpleitk-image-processing',
    'pandas',
    'apoc', # GPU-accelerated pixel and object classification
    'aicsimageio', # for reading .czi files
    'opencv-python',  
    'readlif', # for reading .lif files
    'SimpleITK',
    'openslide-python', # for reading .ndpi files 
    'glob2',
    'pytest',
    'cucim-cu12',
    'cucim',
    'cupy-cuda12x'
]

# Install each dependency using pip
for dependency in dependencies:
    subprocess.call(['pip', 'install', dependency])

# Additional installations for specific packages
subprocess.call(['mamba', 'install', '-y', 
                 'openslide',
                 'ocl-icd-system',
                 'pyopencl','cudatoolkit'])

# install napari

subprocess.call(['python', '-m', 'pip', 'install', 'napari[all]'])

print("All dependencies installed successfully.")
