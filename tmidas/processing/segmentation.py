"""
Segmentation utilities for T-MIDAS.
"""

import numpy as np
from typing import Optional
from skimage.measure import label, regionprops


def label_image(image: np.ndarray) -> np.ndarray:
    """Label connected components in a binary image.
    
    Args:
        image: Binary image array
        
    Returns:
        Labeled image
    """
    return label(image)


def get_region_properties(labeled_image: np.ndarray, intensity_image: Optional[np.ndarray] = None) -> list:
    """Get region properties from labeled image.
    
    Args:
        labeled_image: Labeled image
        intensity_image: Optional intensity image for intensity measurements
        
    Returns:
        List of region properties
    """
    return regionprops(labeled_image, intensity_image)


def filter_small_labels(labeled_image: np.ndarray, min_size: float, output_type: str = 'instance') -> np.ndarray:
    """Remove labels smaller than min_size.
    
    Args:
        labeled_image: Input labeled image
        min_size: Minimum size threshold
        output_type: 'instance' or 'semantic'
        
    Returns:
        Filtered labeled image
    """
    # Create a temporary image where each connected component has a unique ID
    temp_labels = label(labeled_image > 0, connectivity=1)
    
    # Get properties of each object (connected component)
    props = get_region_properties(temp_labels, intensity_image=labeled_image)
    
    # Create an empty array to store the new label image
    new_label_image = np.zeros_like(labeled_image, dtype=labeled_image.dtype)
    
    # Iterate over each object and decide whether to keep it
    for i, prop in enumerate(props, start=1):
        if prop.area > min_size:
            if output_type == 'semantic':
                original_label_id = int(prop.intensity_mean)
                new_label_image[temp_labels == prop.label] = original_label_id
            else:  # instance segmentation
                new_label_image[temp_labels == prop.label] = i
    
    return new_label_image
