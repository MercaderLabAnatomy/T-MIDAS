# Tissue Microscopy Image Data Analysis Suite (T-MIDAS)
T-MIDAS was created with a focus on the reproducibility of typical automated image analysis workflows for biological tissue microscopy data. 
- Executable with a simple text-based user interface
- Runs on any low-grade workstation with a single GPU
  
- Supported imaging modalities:
  - Confocal fluorescence microscopy
  - Wholeslide images
    
- Features include:
  - Image Format Conversion (proprietary to open)
  - Image Preprocessing
  - Image Segmentation
  - Image Segmentation Validation
  - Region-of-interest (ROI) Analyses (see below)
    
- Quick and Easy Installation (see below)


## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/Image_Analysis_Suite/assets/99955854/a2cdf959-501c-431a-958e-09365254edf1)

## List of Features for Automated Batch Processing of images:
User selection 1,1,1 corresponds to  list element 1.1.1

- `1,1,1` Convert .ndpi to .tif
- `1,1,2` Convert .lif to .tif
- `1,1,3` Convert multichannel .czi to .tif
- `1,1,4` Convert brightfield .czi to .tif
- `1,2,1,2,1` Crop heart slices from multichannel .ndpi 
- `1,2,1,2,2` Crop hearts from multichannel .lif image stacks
- `1,2,1,3` Crop heart slices from brightfield .ndpi 
- `1,3` Split multichannel image stack to single channel image stacks (.tif)
- `1,4` Apply maximum intensity projection to image stacks
- `1,5` Tile image 
- `1,6` Sample random tiles
- `2,3` Segment bright spots in 2D
- `2,4` Segment nuclei in 3D
- `2,5` Segment tissue in 3D
- `2,6` Segment myocardium in cropped heart slices
- `3,1` Generate ROIs from heart slice label images (that already contain masks of injury and intact ventricle regions) 
- `3,2` Count spots within ROIs in heart slices
- `3,3` Count nuclei within tissue (3D)
- `4,1` Validate predicted spot counts against manual counts (2D label images)
- `4,2` Validate predicted nuclei counts against manual counts (3D label images)
- `5,1` Analyze multichannel .tif (ndpi viewer exports) of CM culture wells
- `5,2` Count proliferating FITC+ cells

WIP:
- Workflow logging

## Installation

Create conda environment YML (WIP).
