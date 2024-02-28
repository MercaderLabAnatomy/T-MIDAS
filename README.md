![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite


T-MIDAS was created with a focus on the reproducibility of batch image processing workflows for biological tissue microscopy data. 
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
  - Region-of-interest (ROI) Analyses
    
- Quick and Easy Installation

## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/ef71315b-726d-4a2f-9546-d326aba513dd)

![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/a318594b-3cc6-4a67-b7ba-b84860f27266)


## Batch Image Processing 
### List of available scripts
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/868dde8f-2cab-4662-ad60-8880f4ac8c75)

## WIP
- Workflow logging
- Plug & Play DGMs for Transfer Learning (PyTorch framework)
- Lightsheet data
- Time series data
- ML-based hyperparameter selection method for CLAHE

## Installation
```
mamba create -y -n tmidas-env python=3.9
mamba activate tmidas-env
python ./scripts/install_dependencies.py
```
