# T-MIDAS API Documentation

## Overview

T-MIDAS (Tissue Microscopy Image Data Analysis Suite) is a modular Python package for microscopy image processing and analysis. This documentation provides detailed information about the API components.

## Package Structure

```
tmidas/
├── config.py              # Configuration settings
├── processing/            # Domain-specific modules
│   ├── segmentation.py    # Image segmentation utilities
│   └── tracking.py        # Cell tracking utilities
├── utils/                 # Shared utilities
│   ├── io_utils.py        # File I/O operations
│   ├── argparse_utils.py  # CLI argument handling
│   └── env_utils.py       # Environment management
└── tests/                 # Unit tests
```

## Core Modules

### tmidas.config

Configuration settings for T-MIDAS.

**Constants:**
- `TMIDAS_PATH`: Default installation path
- `MODELS_PATH`: Path to pre-trained models
- `DEFAULT_ENV`: Default conda environment name
- `TRACKASTRA_ENV`: TrackAstra environment name
- `DEFAULT_MIN_SIZE`: Default minimum label size
- `DEFAULT_COMPRESSION`: Default TIFF compression

### tmidas.utils.io_utils

Common I/O utilities for image file operations.

#### Functions

##### `read_image(file_path: str) -> Optional[np.ndarray]`

Read an image from file.

**Parameters:**
- `file_path` (str): Path to the image file

**Returns:**
- Image array or None if loading fails

**Example:**
```python
from tmidas.utils.io_utils import read_image
image = read_image("path/to/image.tif")
```

##### `write_image(image: np.ndarray, file_path: str, **kwargs) -> None`

Write an image to file.

**Parameters:**
- `image` (np.ndarray): Image array to save
- `file_path` (str): Path where to save the image
- `**kwargs`: Additional arguments for tifffile.imwrite

**Example:**
```python
from tmidas.utils.io_utils import write_image
write_image(image_array, "output.tif", compression="zlib")
```

##### `get_files_with_extension(directory: str, extension: str) -> list[str]`

Get all files with a specific extension in a directory.

**Parameters:**
- `directory` (str): Directory path
- `extension` (str): File extension (with or without dot)

**Returns:**
- List of filenames

##### `find_matching_files(directory: str, pattern: str) -> list[Path]`

Find files matching a pattern.

**Parameters:**
- `directory` (str): Directory path
- `pattern` (str): Glob pattern

**Returns:**
- List of matching file paths

### tmidas.utils.argparse_utils

Common argument parsing utilities for CLI scripts.

#### Functions

##### `create_parser(description: str, input_help: str = "Path to input folder") -> argparse.ArgumentParser`

Create a basic argument parser.

**Parameters:**
- `description` (str): Description for the parser
- `input_help` (str): Help text for the input argument

**Returns:**
- Configured argument parser

**Example:**
```python
from tmidas.utils.argparse_utils import create_parser
parser = create_parser("Process images", "Path to image folder")
```

##### `add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser`

Add common arguments to parser.

**Parameters:**
- `parser` (ArgumentParser): Argument parser to modify

**Returns:**
- Modified argument parser

### tmidas.utils.env_utils

Environment management utilities.

#### Functions

##### `run_command(command: str, check: bool = True) -> str`

Run a shell command.

**Parameters:**
- `command` (str): Command to execute
- `check` (bool): Whether to raise exception on non-zero exit code

**Returns:**
- Command output

##### `check_environment(env_name: str) -> bool`

Check if a conda environment exists.

**Parameters:**
- `env_name` (str): Name of the environment

**Returns:**
- True if environment exists

##### `create_environment(env_name: str, python_version: str = "3.9") -> None`

Create a conda environment.

**Parameters:**
- `env_name` (str): Name of the environment
- `python_version` (str): Python version to install

##### `install_packages(env_name: str, packages: list[str]) -> None`

Install packages in environment.

**Parameters:**
- `env_name` (str): Name of the environment
- `packages` (list[str]): List of package names

##### `setup_trackastra_env() -> None`

Set up TrackAstra environment with all required dependencies.

### tmidas.processing.segmentation

Image segmentation utilities.

#### Functions

##### `label_image(image: np.ndarray) -> np.ndarray`

Label connected components in a binary image.

**Parameters:**
- `image` (np.ndarray): Binary image array

**Returns:**
- Labeled image

##### `get_region_properties(labeled_image: np.ndarray, intensity_image: Optional[np.ndarray] = None) -> list`

Get region properties from labeled image.

**Parameters:**
- `labeled_image` (np.ndarray): Labeled image
- `intensity_image` (Optional[np.ndarray]): Optional intensity image for intensity measurements

**Returns:**
- List of region properties

##### `filter_small_labels(labeled_image: np.ndarray, min_size: float, output_type: str = 'instance') -> np.ndarray`

Remove labels smaller than min_size.

**Parameters:**
- `labeled_image` (np.ndarray): Input labeled image
- `min_size` (float): Minimum size threshold
- `output_type` (str): 'instance' or 'semantic'

**Returns:**
- Filtered labeled image

**Example:**
```python
from tmidas.processing.segmentation import filter_small_labels
filtered = filter_small_labels(labeled_image, 100.0, 'instance')
```

### tmidas.processing.tracking

Cell tracking utilities.

#### Functions

##### `get_image_mask_pairs(folder: str, label_suffix: str) -> list[Tuple[str, str]]`

Get pairs of images and their corresponding masks.

**Parameters:**
- `folder` (str): Directory containing images and masks
- `label_suffix` (str): Suffix for mask files

**Returns:**
- List of (image_path, mask_path) tuples

##### `track_cells_with_trackastra(img: np.ndarray, mask: np.ndarray, model_name: str = "general_2d", mode: str = "greedy") -> Optional[Any]`

Track cells using TrackAstra.

**Parameters:**
- `img` (np.ndarray): Image array
- `mask` (np.ndarray): Mask array
- `model_name` (str): TrackAstra model name
- `mode` (str): Tracking mode ('greedy', 'ilp', 'greedy_nodiv')

**Returns:**
- Track graph or None if failed

##### `save_tracking_results(track_graph: Any, mask: np.ndarray, output_dir: str, output_format: str = 'ctc') -> Tuple[Optional[Any], Optional[np.ndarray]]`

Save tracking results in specified format.

**Parameters:**
- `track_graph` (Any): Track graph from TrackAstra
- `mask` (np.ndarray): Original mask
- `output_dir` (str): Output directory
- `output_format` (str): Output format ('ctc', 'napari', etc.)

**Returns:**
- Tuple of (tracks, tracked_masks)

## Usage Examples

### Basic Image Processing

```python
import sys
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image
from tmidas.processing.segmentation import filter_small_labels

# Read image
image = read_image("input.tif")

# Process image
filtered = filter_small_labels(image, 50.0)

# Save result
write_image(filtered, "output.tif")
```

### Environment Setup

```python
from tmidas.utils.env_utils import setup_trackastra_env

# Set up TrackAstra environment
setup_trackastra_env()
```

### Cell Tracking

```python
from tmidas.processing.tracking import get_image_mask_pairs, track_cells_with_trackastra

# Get image pairs
pairs = get_image_mask_pairs("data/", "_mask.tif")

# Track cells
for img_path, mask_path in pairs:
    img = read_image(img_path)
    mask = read_image(mask_path)
    track_graph = track_cells_with_trackastra(img, mask, model_name="general_2d")
```

## Error Handling

All functions include proper error handling:
- File I/O operations catch exceptions and return None or print error messages
- Environment operations check for command success
- Import errors are handled gracefully for optional dependencies

## Type Hints

All functions include comprehensive type hints for better IDE support and documentation.

## Dependencies

- numpy
- scikit-image
- tifffile
- tqdm (for progress bars)
- torch (for deep learning models)
- trackastra (for cell tracking)

## Testing

Run tests with:
```bash
python run_tests.py
```

Tests cover:
- I/O operations
- Segmentation utilities
- Error handling
- Type validation
