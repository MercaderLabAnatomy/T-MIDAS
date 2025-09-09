"""
Tests for segmentation utilities.
"""

import numpy as np
import pytest
from tmidas.processing.segmentation import filter_small_labels


def test_filter_small_labels():
    """Test filtering small labels."""
    # Create a test labeled image
    image = np.zeros((10, 10), dtype=np.uint32)
    image[2:4, 2:4] = 1  # Small label (4 pixels)
    image[5:8, 5:8] = 2  # Larger label (9 pixels)
    
    # Filter labels smaller than 5
    result = filter_small_labels(image, 5, 'instance')
    
    # Check that small label is removed
    assert np.sum(result == 1) == 0
    # Check that large label is kept
    assert np.sum(result == 1) == 9 or np.sum(result == 2) == 9


def test_filter_small_labels_semantic():
    """Test filtering small labels with semantic output."""
    image = np.zeros((10, 10), dtype=np.uint32)
    image[2:4, 2:4] = 5  # Small label
    image[5:8, 5:8] = 10  # Larger label
    
    result = filter_small_labels(image, 5, 'semantic')
    
    # Small label should be removed
    assert np.sum(result == 5) == 0
    # Large label should be kept with original ID
    assert np.sum(result == 10) == 9
