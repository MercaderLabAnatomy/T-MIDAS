[![DOI](https://zenodo.org/badge/743431268.svg)](https://zenodo.org/doi/10.5281/zenodo.10728503)



![T-MIDAS Logo](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/aada2d33-f5f7-4395-bf36-c0466b304d0d) 
# Tissue Microscopy Image Data Analysis Suite

Latest news (03/2025): Check out the [T-MIDAS Napari plugin](https://github.com/macromeer/napari-tmidas)!

#### T-MIDAS was created with a focus on the reproducibility of batch image processing and quantification 
- Batch processing pipelines for image format conversion, preprocessing, segmentation, ROI analysis
- Executable with a simple, text-based user interface
- Runs on any low-grade workstation with a single GPU
- Modular and simple codebase with few dependencies for easy maintenance
- Supported imaging modalities: Confocal microscopy, slidescanner, multicolor, brightfield
- Logs all your workflows and your parameter choices to a simple CSV
- You can fork this repository to adapt the batch processing scripts to your own image analysis workflows
- [Quick installation](https://github.com/MercaderLabAnatomy/T-MIDAS?tab=readme-ov-file#installation-ubuntu)

  
T-MIDAS is built on established image processing libraries. All dependencies are listed [here](https://github.com/MercaderLabAnatomy/T-MIDAS/blob/main/scripts/install_dependencies.py). See [acknowledgements and citations](https://github.com/MercaderLabAnatomy/T-MIDAS?tab=readme-ov-file#acknowledgements-and-citations) for further information.

## Text-based User Interface
![image](https://github.com/MercaderLabAnatomy/T-MIDAS/assets/99955854/ef71315b-726d-4a2f-9546-d326aba513dd)

## Feature overview
More detailed information is provided via text-based user interface.
<pre>
[1] Image Preprocessing   
    [1] File Conversion to TIF format
        [1] Convert .ndpi
        [2] Export (selected) series from bioformats-compatible files (.lif,.czi,...)
        [3] Convert brightfield .czi
    [2] Crop largest objects from images using Segment Anything
        [1] Slidescanner images (fluorescent, .ndpi)
        [2] Slidescanner images (brightfield, .ndpi)
        [3] Multicolor image stacks (.lif)
    [3] Extract intersecting regions of two (label) images
    [4] Sample random image subregions (/w random seed)
    [5] Enhance local contrast (CLAHE, 8bit)
    [6] Restore images with Cellpose (single or multiple color channel, 2D or 3D, also time series)
    [7] Split color channels (TIF; 2D or 3D, time series)
    [8] Merge color channels (TIF; 2D or 3D, time series)
    [9] Convert RGB images to Napari label images
    [10] Crop zebrafish larvae from 4x Acquifer images (multicolor but requires brightfield)
    [11] Combine label images
    [12] Remove small labels from label images
    [13] Convert instance label images to binary

[2] Image Segmentation
    [1] Segment bright spots (2D or 3D, time series) 
    [2] Segment blobs (2D or 3D, time series)
        [1] User-defined or automatic (Otsu) thresholding
        [2] Cellpose (cells or nuclei)
    [3] Semantic segmentation (2D/3D, Otsu or manual threshold)
    [4] Human in-the-loop (instance or semantic) segmentation (2D; Segment Anything)
    [5] Improve instance segmentation using CLAHE

[3] Regions of Interest (ROI) Analysis
    [1] Heart slices: Add 100um boundary zone to [intact+injured] ventricle masks
    [2] Count spots within ROI (2D)
    [3] Count blobs within ROI (3D)
    [4] Colocalize ROI in 2 or 3 color channels (counts and sizes)
    [5] Get properties of objects within ROI (two color channels)
    [6] Get basic ROI properties (single channel)
    [7] Get ROI properties in multiclass label images

[4] Cell/Nuclei/Particle Tracking
   [1] Trackastra

[5] Image Segmentation Validation
    [1] Validate spot counts (2D)
    [2] Validate blobs (2D or 3D; global F1 score)

[6] File compression
    [1] Compress files using zstd
    [2] Decompress files using zstd 

[7] Label image inspection with Napari

[n] Start Napari (with useful plugins)
</pre>

## WIP
- AI for ROI detection in brightfield and fluorescence images
- Code stability


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

### Napari T-MIDAS plugin
If you want to use T-MIDAS in Napari, please follow [these installation instructions](https://github.com/macromeer/napari-tmidas?tab=readme-ov-file#installation) for the plugin. Currently, you have to install it in its own virtual environment (call it `napari-tmidas`).

## Usage
To start the text-based user interface in your terminal, change directory to the T-MIDAS folder and type 
```
python ./scripts/user_welcome.py
```

## Acknowledgements and Citations

This project relies on several open-source libraries and tools. We would like to acknowledge and thank the creators of these projects:

### Core Libraries

- [NumPy](https://numpy.org/): Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2

- [CuPy](https://github.com/cupy/cupy): Nishino, R. O. Y. U. D., & Loomis, S. H. C. (2017). Cupy: A numpy-compatible library for nvidia gpu calculations. 31st confernce on neural information processing systems, 151(7). 

- [scikit-image](https://scikit-image.org/): van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

- [tifffile](https://pypi.org/project/tifffile/): Gohlke, C. (2021). tifffile (Version X.X.X) [Software]. Available from https://pypi.org/project/tifffile/

- [pyclesperanto-prototype](https://github.com/clEsperanto/pyclesperanto_prototype): Haase, R., Royer, L. A., Steinbach, P., Schmidt, D., Dibrov, A., Schmidt, U., ... & Myers, E. W. (2020). CLIJ: GPU-accelerated image processing for everyone. Nature methods, 17(1), 5-6.

- [Pillow](https://python-pillow.org/): Clark, A. (2015). Pillow (PIL Fork) Documentation. readthedocs.

- [pandas](https://pandas.pydata.org/): McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56).

- [OpenCV](https://opencv.org/): Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

- [SimpleITK](https://simpleitk.org/): Lowekamp, B. C., Chen, D. T., Ibáñez, L., & Blezek, D. (2013). The design of SimpleITK. Frontiers in neuroinformatics, 7, 45.

- [PyTorch](https://pytorch.org/): Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8026-8037).

### Image Analysis and Processing

- [Cellpose](https://github.com/MouseLand/cellpose): Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106.

- [napari](https://napari.org/): Sofroniew, N., Lambert, T., Evans, K., Nunez-Iglesias, J., Yamauchi, K., Solak, A. C., ... & Eliceiri, K. (2022). napari: a multi-dimensional image viewer for python. Zenodo.

- [apoc](https://github.com/haesleinhuepf/apoc): Haase, R., Royer, L. A., Steinbach, P., Schmidt, D., Dibrov, A., Schmidt, U., ... & Myers, E. W. (2020). CLIJ: GPU-accelerated image processing for everyone. Nature methods, 17(1), 5-6.

- [aicsimageio](https://github.com/AllenCellModeling/aicsimageio): Mancini, M., Colon-Hernandez, P., & Amodaj, N. (2020). AICSImageIO: A Python library for reading and writing image data. Journal of Open Source Software, 5(55), 2584.

- [readlif](https://github.com/nimne/readlif): Nimne. (2021). readlif: Python package to read Leica LIF files. GitHub repository.

- [OpenSlide](https://openslide.org/): Goode, A., Gilbert, B., Harkes, J., Jukic, D., & Satyanarayanan, M. (2013). OpenSlide: A vendor-neutral software foundation for digital pathology. Journal of pathology informatics, 4.

- [cucim](https://github.com/rapidsai/cucim): RAPIDS Team. (2021). cuCIM: GPU accelerated image processing. GitHub repository.

- [aicspylibczi](https://github.com/AllenCellModeling/aicspylibczi): Allen Institute for Cell Science. (2021). aicspylibczi: Python bindings for libCZI. GitHub repository.

- [python-bioformats](https://github.com/CellProfiler/python-bioformats): Carpenter, A. E., Jones, T. R., Lamprecht, M. R., Clarke, C., Kang, I. H., Friman, O., ... & Sabatini, D. M. (2006). CellProfiler: image analysis software for identifying and quantifying cell phenotypes. Genome biology, 7(10), R100.

### Deep Learning Models

- [torchvision](https://github.com/pytorch/vision): Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems (pp. 8026-8037).

- [timm](https://github.com/rwightman/pytorch-image-models): Wightman, R. (2019). PyTorch Image Models. GitHub repository.

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM): Zhang, C., Han, D., Qiao, Y., Kim, J. U., Bae, S. H., Lee, S., & Hong, C. S. (2023). Faster Segment Anything: Towards Lightweight SAM for Mobile Applications. arXiv preprint arXiv:2306.14289.

### Additional Tools

- [tqdm](https://github.com/tqdm/tqdm): da Costa-Luis, C., Larroque, S. K., Altendorf, K., Mary, H., richardsheridan, Korobov, M., ... & Trofimov, A. (2022). tqdm: A Fast, Extensible Progress Bar for Python and CLI. Zenodo.

- [pytest](https://docs.pytest.org/): Krekel, H., Oliveira, B., Pfannschmidt, R., Bruynooghe, F., Laugher, B., & Bruhin, F. (2004). pytest: helps you write better programs. The pytest Development Team.

We are grateful to the developers and maintainers of these projects for their valuable contributions to the open-source community.

