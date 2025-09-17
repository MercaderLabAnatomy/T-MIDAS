"""
Comprehensive script testing for T-MIDAS.
Tests multiple scripts across different categories.
"""

import os
import tempfile
import shutil
import subprocess
import sys
import numpy as np
import pytest

# Add tmidas to path
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image


def create_test_data():
    """Create comprehensive test data for various script types."""
    test_dir = tempfile.mkdtemp()

    # Create sample image
    sample_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    image_path = os.path.join(test_dir, "sample.tif")
    write_image(sample_image, image_path)

    # Create sample labels
    labels = np.zeros((100, 100), dtype=np.uint32)
    labels[10:30, 10:30] = 1  # 400 pixels
    labels[50:70, 50:70] = 2  # 400 pixels
    labels[80:85, 80:85] = 3  # 25 pixels (small)
    labels_path = os.path.join(test_dir, "sample_labels.tif")
    write_image(labels, labels_path)

    # Create multi-channel image
    multichannel = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    multichannel_path = os.path.join(test_dir, "multichannel.tif")
    write_image(multichannel, multichannel_path)

    # Create 3D image stack
    stack_3d = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)
    stack_path = os.path.join(test_dir, "stack.tif")
    write_image(stack_3d, stack_path)

    return test_dir


def run_script_test(script_name, args, test_dir, timeout=30):
    """Run a script test with given arguments."""
    script_path = f"/opt/T-MIDAS/scripts/{script_name}"

    if not os.path.exists(script_path):
        print(f"Script {script_name} not found")
        return False

    try:
        cmd = [sys.executable, script_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=test_dir)

        if result.returncode != 0:
            print(f"Script {script_name} failed: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"Script {script_name} timed out")
        return False
    except Exception as e:
        print(f"Script {script_name} error: {e}")
        return False


class TestFileConversionScripts:
    """Test file conversion scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_split_color_channels(self):
        """Test splitting color channels."""
        args = ["--input", self.test_dir, "--pattern", "multichannel.tif"]
        success = run_script_test("split_color_channels.py", args, self.test_dir)
        assert success, "split_color_channels script failed"

    def test_merge_color_channels(self):
        """Test merging color channels."""
        # First split, then merge
        split_args = ["--input", self.test_dir, "--pattern", "multichannel.tif"]
        run_script_test("split_color_channels.py", split_args, self.test_dir)

        merge_args = ["--input", self.test_dir, "--pattern", "_ch", "--output", "merged.tif"]
        success = run_script_test("merge_color_channels.py", merge_args, self.test_dir)
        assert success, "merge_color_channels script failed"

    def test_convert_instance_to_semantic(self):
        """Test instance to semantic conversion."""
        args = ["--input", self.test_dir, "--pattern", "_labels.tif"]
        success = run_script_test("convert_instance_to_semantic.py", args, self.test_dir)
        assert success, "convert_instance_to_semantic script failed"


class TestSegmentationScripts:
    """Test segmentation-related scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_segmentation_semantic(self):
        """Test semantic segmentation."""
        args = ["--input", self.test_dir, "--method", "otsu"]
        success = run_script_test("segmentation_semantic.py", args, self.test_dir)
        assert success, "segmentation_semantic script failed"

    def test_segmentation_blobs(self):
        """Test blob segmentation."""
        args = ["--input", self.test_dir, "--method", "otsu"]
        success = run_script_test("segmentation_blobs.py", args, self.test_dir)
        assert success, "segmentation_blobs script failed"

    def test_segmentation_spots(self):
        """Test spot segmentation."""
        args = ["--input", self.test_dir]
        success = run_script_test("segmentation_spots.py", args, self.test_dir)
        assert success, "segmentation_spots script failed"


class TestROIScripts:
    """Test ROI analysis scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_roi_count_instances_2d(self):
        """Test 2D instance counting."""
        args = ["--input", self.test_dir, "--label_pattern", "_labels.tif"]
        success = run_script_test("ROI_count_instances_2D.py", args, self.test_dir)
        assert success, "ROI_count_instances_2D script failed"

    def test_roi_count_instances_3d(self):
        """Test 3D instance counting."""
        args = ["--input", self.test_dir, "--label_pattern", "_labels.tif"]
        success = run_script_test("ROI_count_instances_3D.py", args, self.test_dir)
        assert success, "ROI_count_instances_3D script failed"

    def test_get_class_regionprops(self):
        """Test class region properties."""
        args = ["--input", self.test_dir, "--label_pattern", "_labels.tif"]
        success = run_script_test("get_class_regionprops.py", args, self.test_dir)
        assert success, "get_class_regionprops script failed"


class TestImageProcessingScripts:
    """Test image processing scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_deep_tissue_clahe(self):
        """Test CLAHE enhancement."""
        args = ["--input", self.test_dir]
        success = run_script_test("deep_tissue_clahe.py", args, self.test_dir)
        assert success, "deep_tissue_clahe script failed"

    def test_combine_labels(self):
        """Test label combination."""
        args = ["--input", self.test_dir, "--pattern", "_labels.tif", "--output", "combined.tif"]
        success = run_script_test("combine_labels.py", args, self.test_dir)
        assert success, "combine_labels script failed"

    def test_export_label(self):
        """Test label export."""
        args = ["--input", self.test_dir, "--pattern", "_labels.tif"]
        success = run_script_test("export_label.py", args, self.test_dir)
        assert success, "export_label script failed"


class TestUtilityScripts:
    """Test utility scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_random_tile_sampler(self):
        """Test random tile sampling."""
        args = ["--input", self.test_dir, "--tile_size", "50", "--num_tiles", "5"]
        success = run_script_test("random_tile_sampler.py", args, self.test_dir)
        assert success, "random_tile_sampler script failed"

    def test_intersection(self):
        """Test image intersection."""
        args = ["--input", self.test_dir, "--pattern1", ".tif", "--pattern2", "_labels.tif"]
        success = run_script_test("intersection.py", args, self.test_dir)
        assert success, "intersection script failed"


class TestCompressionScripts:
    """Test compression/decompression scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_zstd_compression(self):
        """Test ZSTD compression."""
        args = ["--input", self.test_dir, "--pattern", ".tif"]
        success = run_script_test("zstd_compression.py", args, self.test_dir)
        assert success, "zstd_compression script failed"

    def test_zstd_decompression(self):
        """Test ZSTD decompression."""
        # First compress
        compress_args = ["--input", self.test_dir, "--pattern", ".tif"]
        run_script_test("zstd_compression.py", compress_args, self.test_dir)

        # Then decompress
        decompress_args = ["--input", self.test_dir, "--pattern", ".zst"]
        success = run_script_test("zstd_decompression.py", decompress_args, self.test_dir)
        assert success, "zstd_decompression script failed"


class TestValidationScripts:
    """Test validation scripts."""

    def setup_method(self):
        self.test_dir = create_test_data()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_counts_validation(self):
        """Test counts validation."""
        args = ["--input", self.test_dir, "--pattern", "_labels.tif"]
        success = run_script_test("counts_validation.py", args, self.test_dir)
        assert success, "counts_validation script failed"

    def test_segmentation_validation_f1_score(self):
        """Test segmentation validation with F1 score."""
        args = ["--input", self.test_dir, "--pattern", "_labels.tif"]
        success = run_script_test("segmentation_validation_f1_score.py", args, self.test_dir)
        assert success, "segmentation_validation_f1_score script failed"


def test_script_syntax_validation():
    """Test that all scripts have valid Python syntax."""
    scripts_dir = "/opt/T-MIDAS/scripts"
    failed_scripts = []

    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            script_path = os.path.join(scripts_dir, filename)
            try:
                import py_compile
                py_compile.compile(script_path, doraise=True)
            except py_compile.PyCompileError as e:
                failed_scripts.append((filename, str(e)))

    if failed_scripts:
        failure_msg = "Scripts with syntax errors:\n"
        for script, error in failed_scripts:
            failure_msg += f"  {script}: {error}\n"
        pytest.fail(failure_msg)


def test_script_import_validation():
    """Test that all scripts can import their dependencies."""
    scripts_dir = "/opt/T-MIDAS/scripts"
    failed_scripts = []

    for filename in os.listdir(scripts_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            script_path = os.path.join(scripts_dir, filename)
            try:
                # Try to import the script as a module
                spec = importlib.util.spec_from_file_location(filename[:-3], script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as e:
                failed_scripts.append((filename, str(e)))

    if failed_scripts:
        failure_msg = "Scripts with import errors:\n"
        for script, error in failed_scripts:
            failure_msg += f"  {script}: {error}\n"
        pytest.fail(failure_msg)


if __name__ == "__main__":
    # Run syntax and import validation
    print("Running comprehensive script validation...")

    # Syntax check
    print("Checking syntax...")
    try:
        test_script_syntax_validation()
        print("✓ All scripts have valid syntax")
    except Exception as e:
        print(f"❌ Syntax errors found: {e}")

    # Import check
    print("Checking imports...")
    try:
        import importlib.util
        test_script_import_validation()
        print("✓ All scripts can import dependencies")
    except Exception as e:
        print(f"❌ Import errors found: {e}")

    print("Comprehensive validation complete!")
