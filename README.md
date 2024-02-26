# A Tiny Bioimage Analysis Suite
- A collection of Python scripts exectuable with a simple text-based user interface. 
- For automated processing and analysis of biological tissue images.
- Runs on any low-grade workstation with a single GPU. 
- Supported imaging modalities:
  - Confocal fluorescence microscopy (2D, 3D).
  - Wholeslide images (.czi, .ndpi).
Below you can find a list of currently available features.

## Text-based user interface via CLI / SSH
![image](https://github.com/MercaderLabAnatomy/Image_Analysis_Suite/assets/99955854/a2cdf959-501c-431a-958e-09365254edf1)

## List of features for automated batch processing of images:
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
