![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite


#### T-MIDAS was created with a focus on the reproducibility of batch image processing 
- Format conversion, preprocessing, segmentation, ROI analyses
- Executable with a simple, text-based user interface
- Runs on any low-grade workstation with a single GPU
- Modular and simple codebase with few dependencies for easy maintenance
- Quick installation
- Supported imaging modalities: Confocal microscopy, slidescanner, multicolor, brightfield
    
    


## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/ef71315b-726d-4a2f-9546-d326aba513dd)

## Batch Image Processing
<pre>
[1] Image Preprocessing
    [1] File Conversion to TIFF
        [1] Convert .ndpi
        [2] Convert .lif
        [3] Convert brightfield .czi
    [2] Cropping Blobs from Images
        [1] Slidescanner images (fluorescent, .ndpi)
        [2] Slidescanner images (brightfield, .ndpi)
        [3] Multicolor image stacks (.lif)
    [3] Sample Random Image Subregions
    [4] Normalize intensity across image (CLAHE)
[2] Image Segmentation
    [1] Segment bright spots (2D)
    [2] Segment blobs (3D; requires dark background and good SNR)
    [3] Semantic segmentation (2D, fluorescence or brightfield)
    [4] Semantic segmentation (3D; requires dark background and good SNR)
[3] Regions of Interest (ROI) Analysis
    [1] Heart slices: Generate ROIs from [intact+injured] ventricle masks
    [2] Heart slices: Count spots within ventricle ROIs
    [3] Heart volume: Count nuclei within ROIs
[4] Image Segmentation Validation
    [1] Validate predicted counts against manual counts (2D label images)
    [2] Validate predicted segmentation results against manual segmentation results (2D or 3D label images)   
</pre>

## WIP
- Workflow logging
- Plug & Play DGMs for Transfer Learning (PyTorch framework)
- Lightsheet data
- Time series data
- ML-based hyperparameter selection method for CLAHE
- Multicolor CLAHE

## Installation
A prerequisite is the [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) package and environment management system. 
The minimal Conda installer [miniforge](https://github.com/conda-forge/miniforge) is preferable for its simplicity and speed. 
After installing miniforge, you can use the miniforge terminal in Windows and macOS operating systems (Linux already has a terminal).
In the terminal, run the following commands to create and activate a conda environment for T-MIDAS:
```
mamba create -y -n tmidas-env python=3.9
mamba activate tmidas-env
```
Now you need to download the T-MIDAS repository either using `git clone https://github.com/MercaderLabAnatomy/T-MIDAS.git` or by downloading and unpacking the [ZIP](https://github.com/MercaderLabAnatomy/T-MIDAS/archive/refs/heads/main.zip). In your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/install_dependencies.py
```
## Usage
To start the text-based user interface in your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/user_welcome.py`
```
