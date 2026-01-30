"""
Common I/O utilities for T-MIDAS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Any
from skimage.io import imread
from tifffile import imwrite
import numpy as np


def read_image(file_path: str) -> Optional[np.ndarray]:
    """Read an image from file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image array or None if loading fails
    """
    try:
        return imread(file_path)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def write_image(image: np.ndarray, file_path: str, **kwargs: Any) -> None:
    """Write an image to file.
    
    Args:
        image: Image array to save
        file_path: Path where to save the image
        **kwargs: Additional arguments for tifffile.imwrite
    """
    try:
        if 'compression' not in kwargs:
            kwargs['compression'] = 'zlib'
        imwrite(file_path, image.astype(np.uint32), **kwargs)
    except Exception as e:
        print(f"Error saving image {file_path}: {e}")


def get_files_with_extension(directory: str, extension: str) -> list[str]:
    """Get all files with a specific extension in a directory.
    
    Args:
        directory: Directory path
        extension: File extension (with or without dot)
        
    Returns:
        List of filenames
    """
    return [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]


def find_matching_files(directory: str, pattern: str) -> list[Path]:
    """Find files matching a pattern.
    
    Args:
        directory: Directory path
        pattern: Glob pattern
        
    Returns:
        List of matching file paths
    """
    return list(Path(directory).glob(pattern))
