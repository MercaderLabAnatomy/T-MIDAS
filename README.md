![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite


T-MIDAS was created with a focus on the reproducibility of typical automated image analysis workflows for biological tissue microscopy data. 
- Executable with a simple text-based user interface
- Runs on any low-grade workstation with a single GPU
  
- Supported imaging modalities:
  - Confocal microscopy images
  - Slidescanner images
  - Multicolor and brightfield
    
- Features include:
  - Image Format Conversion (proprietary to open)
  - Image Preprocessing
  - Image Segmentation
  - Image Segmentation Validation
  - Region-of-interest (ROI) Analyses (see below)
    
- Quick and Easy Installation (see below)

## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/f507d524-c991-4c04-b4d8-4e497de50f83)

## List of Features for Automated Batch Processing

![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/b40bf6fd-23f9-4a8a-a7a9-91225e5e5d99)


## WIP
- Workflow logging
- Plug & Play DGMs for Transfer Learning (PyTorch framework)
- Lightsheet data
- Time series data

## Installation
`mamba env create -f tmidas-env.yml`
