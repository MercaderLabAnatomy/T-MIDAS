"""
Tests for I/O utilities.
"""

import os
import tempfile
import numpy as np
from tmidas.utils.io_utils import read_image, write_image


def test_read_write_image():
    """Test reading and writing images."""
    # Create a test image
    test_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        # Write image
        write_image(test_image, tmp.name)
        
        # Read image
        loaded_image = read_image(tmp.name)
        
        # Check that images are equal
        np.testing.assert_array_equal(test_image, loaded_image)
        
        # Clean up
        os.unlink(tmp.name)
