[![DOI](https://zenodo.org/badge/743431268.svg)](https://zenodo.org/doi/10.5281/zenodo.10728503)

![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite


#### T-MIDAS was created with a focus on the reproducibility of batch image processing 
- Image format conversion, preprocessing, segmentation, ROI analysis
- Executable with a simple, text-based user interface
- Runs on any low-grade workstation with a single GPU
- Modular and simple codebase with few dependencies for easy maintenance
- Supported imaging modalities: Confocal microscopy, slidescanner, multicolor, brightfield
- Logs all your workflows and your parameter choices to a simple CSV
- You can fork this repository to adapt the batch processing scripts to your own image analysis workflows
- [Quick installation](https://github.com/MercaderLabAnatomy/T-MIDAS?tab=readme-ov-file#installation)

  
T-MIDAS is built on established image processing libraries such as [scikit-image](https://github.com/scikit-image/scikit-image) or [py-clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype). 

All dependencies are listed [here](https://github.com/MercaderLabAnatomy/T-MIDAS/blob/main/scripts/install_dependencies.py). 

## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/ef71315b-726d-4a2f-9546-d326aba513dd)

## Batch Image Processing
<pre>
[1] Image Preprocessing
    [1] File Conversion to TIFF
        [1] Convert .ndpi
        [2] Convert .lif
        [3] Convert brightfield .czi
    [2] Cropping Largest Objects from Images
        [1] Slidescanner images (fluorescent, .ndpi)
        [2] Slidescanner images (brightfield, .ndpi)
        [3] Multicolor image stacks (.lif)
    [3] Extract Blob Region from Images
    [4] Sample Random Image Subregions
    [5] Normalize intensity across single color image (CLAHE)
[2] Image Segmentation
    [1] Segment bright spots (2D)
    [2] Segment blobs (3D; requires dark background and good SNR)
    [3] Semantic segmentation (2D, fluorescence or brightfield)
    [4] Semantic segmentation (3D; requires dark background and good SNR)
    [5] Segment CLAHE'd images
    [6] Segment multicolor images of cell cultures (2D)
[3] Regions of Interest (ROI) Analysis
    [1] Heart slices: Generate ROIs from [intact+injured] ventricle masks
    [2] Heart slices: Count spots within ventricle ROIs
    [3] Heart volume: Count nuclei within ROIs
    [4] Colocalize ROIs (e.g. nuclei and cell bodies)
[4] Image Segmentation Validation
    [1] Validate predicted counts against manual counts (2D label images)
    [2] Validate predicted segmentation results against manual segmentation results (2D or 3D label images)   
</pre>

## WIP
- Plug & Play DGMs for Transfer Learning (PyTorch framework)
- Lightsheet data
- Time series data


## Installation (Ubuntu)
A prerequisite is the [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) package and environment management system. 
The minimal Conda installer [miniforge](https://github.com/conda-forge/miniforge) is preferable for its simplicity and speed. 
After installing miniforge, you can use `mamba` in the Linux terminal.
In the terminal, run the following commands to create and activate a conda environment for T-MIDAS:
```
mamba create -y -n tmidas-env python=3.9
mamba activate tmidas-env
```
Now you need to download the T-MIDAS repository either using 
```
git clone https://github.com/MercaderLabAnatomy/T-MIDAS.git
```
or by downloading and unpacking the [ZIP](https://github.com/MercaderLabAnatomy/T-MIDAS/archive/refs/heads/main.zip). In your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/install_dependencies.py
```
## Usage
To start the text-based user interface in your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/user_welcome.py
```
