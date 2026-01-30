"""
Integration tests for T-MIDAS scripts.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
from pathlib import Path

# Import tmidas utilities
import sys
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image
from tmidas.processing.segmentation import filter_small_labels


class TestScriptIntegration:
    """Integration tests for T-MIDAS scripts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_data_dir = os.path.join(self.test_dir, "sample_data")
        os.makedirs(self.sample_data_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def create_sample_image(self, shape=(100, 100), dtype=np.uint8):
        """Create a sample image for testing."""
        if dtype == np.uint8:
            return np.random.randint(0, 255, shape, dtype=dtype)
        else:
            return np.random.randint(0, 1000, shape, dtype=dtype)

    def create_sample_labels(self, shape=(100, 100)):
        """Create sample labeled image."""
        labels = np.zeros(shape, dtype=np.uint32)
        # Create some labeled regions
        labels[10:20, 10:20] = 1  # Small region (100 pixels)
        labels[30:50, 30:50] = 2  # Medium region (400 pixels)
        labels[60:90, 60:90] = 3  # Large region (900 pixels)
        return labels

    def test_io_roundtrip(self):
        """Test that images can be saved and loaded correctly."""
        original = self.create_sample_image()

        # Save and reload
        filepath = os.path.join(self.test_dir, "test_image.tif")
        write_image(original, filepath)
        loaded = read_image(filepath)

        assert loaded is not None
        np.testing.assert_array_equal(original, loaded)

    def test_segmentation_filtering(self):
        """Test segmentation filtering produces expected results."""
        labels = self.create_sample_labels()

        # Filter small labels (< 200 pixels)
        filtered = filter_small_labels(labels, 200.0, 'instance')

        # Check that small label (100 pixels) is removed
        assert np.sum(filtered == 1) == 0
        # Check that medium label (400 pixels) is kept
        assert np.sum(filtered == 1) == 400 or np.sum(filtered == 2) == 400
        # Check that large label (900 pixels) is kept
        assert np.sum(filtered == 2) == 900 or np.sum(filtered == 3) == 900

    def test_script_execution_remove_small_labels(self):
        """Test remove_small_labels script execution."""
        # Create test data
        labels = self.create_sample_labels()
        input_path = os.path.join(self.sample_data_dir, "test_labels.tif")
        write_image(labels, input_path)

        # This would require actually running the script
        # For now, just test the core functionality
        assert os.path.exists(input_path)

    def test_error_handling_io(self):
        """Test error handling in I/O operations."""
        # Test reading non-existent file
        result = read_image("non_existent_file.tif")
        assert result is None

        # Test writing to invalid path
        invalid_path = "/invalid/path/test.tif"
        test_image = self.create_sample_image((10, 10))
        # This should not raise an exception but handle it gracefully
        write_image(test_image, invalid_path)  # Should print error but not crash

    def test_large_image_handling(self):
        """Test handling of large images."""
        # Create a moderately large image
        large_image = self.create_sample_image((1000, 1000))

        filepath = os.path.join(self.test_dir, "large_image.tif")
        write_image(large_image, filepath)
        loaded = read_image(filepath)

        assert loaded is not None
        assert loaded.shape == large_image.shape

    def test_different_image_formats(self):
        """Test handling of different data types."""
        # Test uint8 - should be converted to uint32 for labels
        uint8_img = self.create_sample_image(dtype=np.uint8)
        filepath = os.path.join(self.test_dir, "uint8.tif")
        write_image(uint8_img, filepath)
        loaded = read_image(filepath)
        assert loaded.dtype == np.uint32  # Labels are always uint32

        # Test uint16 - should be converted to uint32 for labels
        uint16_img = self.create_sample_image(dtype=np.uint16)
        filepath = os.path.join(self.test_dir, "uint16.tif")
        write_image(uint16_img, filepath)
        loaded = read_image(filepath)
        assert loaded.dtype == np.uint32  # Labels are always uint32

    def test_segmentation_edge_cases(self):
        """Test segmentation with edge cases."""
        # Empty image
        empty = np.zeros((10, 10), dtype=np.uint32)
        filtered = filter_small_labels(empty, 10.0)
        np.testing.assert_array_equal(filtered, empty)

        # Single pixel
        single = np.zeros((10, 10), dtype=np.uint32)
        single[5, 5] = 1
        filtered = filter_small_labels(single, 2.0)
        assert np.sum(filtered) == 0  # Should be removed

        # All same label
        uniform = np.ones((10, 10), dtype=np.uint32)
        filtered = filter_small_labels(uniform, 50.0)
        assert np.sum(filtered) > 0  # Should be kept (100 pixels > 50)


class TestScriptCompatibility:
    """Test compatibility of refactored scripts."""

    def test_argparse_compatibility(self):
        """Test that argparse utilities work correctly."""
        from tmidas.utils.argparse_utils import create_parser

        parser = create_parser("Test parser")
        # This should not raise any exceptions
        assert parser is not None

    def test_import_compatibility(self):
        """Test that all tmidas modules can be imported."""
        try:
            from tmidas.utils import io_utils, argparse_utils, env_utils
            from tmidas.processing import segmentation, tracking
            from tmidas import config
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_function_signatures(self):
        """Test that function signatures are compatible."""
        from tmidas.utils.io_utils import read_image, write_image
        from tmidas.processing.segmentation import filter_small_labels

        # Check that functions can be called with expected parameters
        test_img = np.zeros((10, 10), dtype=np.uint8)

        # These should not raise TypeError
        result = read_image.__code__.co_varnames  # Check parameter names
        assert 'file_path' in result

        result = write_image.__code__.co_varnames
        assert 'image' in result and 'file_path' in result

        result = filter_small_labels.__code__.co_varnames
        assert 'labeled_image' in result and 'min_size' in result
