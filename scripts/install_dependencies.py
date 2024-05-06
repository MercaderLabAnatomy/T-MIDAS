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
    'cucim',
    'aicspylibczi',
    'torch',
    'torchvision',
    'timm'
    
]


# first create and activate mamba environment
subprocess.call(['mamba', 'create', '-y', '-n', 'tmidas-env', 'python=3.8'])
subprocess.call(['mamba', 'activate', 'tmidas-env'])
# init mamba environment
subprocess.call(['mamba', 'init'])


# proceed with installation of dependencies

subprocess.call(['python', '-m', 'pip', 'install', '-U', 'setuptools', 'pip'])

# Additional installations for specific packages
subprocess.call(['mamba', 'install', '-y', 
                 'openslide',
                 'ocl-icd-system',
                 'pyopencl',
                 'cupy']) # if not installed with mamba but pip: gives error with cublas


subprocess.call(['pip', 'install', 'git+https://github.com/ChaoningZhang/MobileSAM.git'])

for dependency in dependencies:
    subprocess.call(['pip', 'install', dependency])


# install napari

subprocess.call(['python', '-m', 'pip', 'install', 'napari[all]'])

# install cellpose
subprocess.call(['python', '-m', 'pip', 'install', 'cellpose'])

# deactivate mamba environment
subprocess.call(['mamba', 'deactivate'])



print("All dependencies installed successfully.")
