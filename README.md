[![DOI](https://zenodo.org/badge/743431268.svg)](https://zenodo.org/doi/10.5281/zenodo.10728503)

![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite


#### T-MIDAS was created with a focus on the reproducibility of batch image processing and quantification 
- Batch processing pipelines for image format conversion, preprocessing, segmentation, ROI analysis
- Executable with a simple, text-based user interface
- Runs on any low-grade workstation with a single GPU
- Modular and simple codebase with few dependencies for easy maintenance
- Supported imaging modalities: Confocal microscopy, slidescanner, multicolor, brightfield
- Logs all your workflows and your parameter choices to a simple CSV
- You can fork this repository to adapt the batch processing scripts to your own image analysis workflows
- [Quick installation](https://github.com/MercaderLabAnatomy/T-MIDAS?tab=readme-ov-file#installation-ubuntu)

  
T-MIDAS is built on established image processing libraries such as [scikit-image](https://github.com/scikit-image/scikit-image), [py-clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype) and [CuPy](https://github.com/cupy/cupy). 

All dependencies are listed [here](https://github.com/MercaderLabAnatomy/T-MIDAS/blob/main/scripts/install_dependencies.py). 

See selected [references](https://github.com/MercaderLabAnatomy/T-MIDAS?tab=readme-ov-file#references) for more information.

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
    [6] Split color channels
    [7] Merge color channels
[2] Image Segmentation
    [1] Segment bright spots (2D)
    [2] Segment blobs (2D)
        [1] Gauss-Otsu-labeling
        [2] Cellpose's (generalist) cyto3 model
    [3] Segment blobs (3D; requires dark background and good SNR)
    [4] Semantic segmentation (2D; fluorescence or brightfield)
    [5] Semantic segmentation (2D; Segment Anything)
    [6] Semantic segmentation (3D; requires dark background and good SNR)
    [7] Improve instance segmentation using CLAHE
[3] Regions of Interest (ROI) Analysis
    [1] Heart slices: Generate ROI from [intact+injured] ventricle masks
    [2] Count spots within ROI (2D)
    [3] Count blobs within ROI (3D)
    [4] Count Colocalization of ROI in 2 or 3 color channels
    [5] Get properties of objects within ROI (two channels)
    [6] Get basic ROI properties (single channel)
[4] Image Segmentation Validation
    [1] Validate spot counts (2D)
    [2] Validate blobs (2D or 3D; global F1 score)   
</pre>

## WIP
- Plug & Play DGMs for Transfer Learning (PyTorch framework)
- Lightsheet data
- Time series data


## Installation (Ubuntu)
A prerequisite is the [Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) package and environment management system. 
The minimal Conda installer [miniforge](https://github.com/conda-forge/miniforge) is preferable for its simplicity and speed. 
After installing miniforge, you can use `mamba` in the Linux terminal. Now you need to download the T-MIDAS repository either using 
```
git clone https://github.com/MercaderLabAnatomy/T-MIDAS.git
```
or by downloading and unpacking the [ZIP](https://github.com/MercaderLabAnatomy/T-MIDAS/archive/refs/heads/main.zip). In your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/install_dependencies.py
```
This will create the T-MIDAS environment and install all its dependencies.
## Usage
To start the text-based user interface in your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/user_welcome.py
```
## References

- [scikit-image processing toolbox documentation](https://scikit-image.org/docs/stable/)
- [py-clesperanto documentation](https://clesperanto.github.io/pyclesperanto/index.html)
- [Cellpose Documentation](https://cellpose.readthedocs.io)
- [Bio-image Analysis Notebooks](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks)
