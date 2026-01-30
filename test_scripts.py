"""
Script execution tests to verify refactored scripts work correctly.
"""

import os
import tempfile
import shutil
import subprocess
import sys
import numpy as np

# Add tmidas to path
sys.path.insert(0, '/opt/T-MIDAS')
from tmidas.utils.io_utils import read_image, write_image


def create_test_data():
    """Create test data for script testing."""
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

    return test_dir


def test_remove_small_labels_script():
    """Test the remove_small_labels script execution."""
    test_dir = create_test_data()

    try:
        # Run the script
        cmd = [
            sys.executable,
            "/opt/T-MIDAS/scripts/remove_small_labels.py",
            "--input", test_dir,
            "--label_suffix", "_labels.tif",
            "--min_size", "100",
            "--output_type", "instance"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"Script failed with error: {result.stderr}")
            return False

        # Check if output file was created
        output_path = os.path.join(test_dir, "sample_labels.tif")
        if not os.path.exists(output_path):
            print("Output file was not created")
            return False

        # Verify the output
        output_labels = read_image(output_path)
        if output_labels is None:
            print("Could not read output labels")
            return False

        # Check that small labels were removed (label 3 should be gone)
        unique_labels = np.unique(output_labels)
        if 3 in unique_labels:
            print("Small label was not removed")
            return False

        print("✓ remove_small_labels script test passed")
        return True

    except subprocess.TimeoutExpired:
        print("Script timed out")
        return False
    except Exception as e:
        print(f"Script test failed with exception: {e}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_get_basic_regionprops_script():
    """Test the get_basic_regionprops script execution."""
    test_dir = create_test_data()

    try:
        # Run the script
        cmd = [
            sys.executable,
            "/opt/T-MIDAS/scripts/get_basic_regionprops.py",
            "--input", test_dir,
            "--label_pattern", "_labels.tif",
            "--channel", "-1"  # No intensity channel
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"Script failed with error: {result.stderr}")
            return False

        # Check if CSV was created
        csv_path = os.path.join(test_dir, "regionprops.csv")
        if not os.path.exists(csv_path):
            print("CSV output file was not created")
            return False

        # Check CSV content
        with open(csv_path, 'r') as f:
            content = f.read()
            if len(content.strip()) == 0:
                print("CSV file is empty")
                return False

        print("✓ get_basic_regionprops script test passed")
        return True

    except subprocess.TimeoutExpired:
        print("Script timed out")
        return False
    except Exception as e:
        print(f"Script test failed with exception: {e}")
        return False
    finally:
        shutil.rmtree(test_dir)


def test_segmentation_sam_script():
    """Test the segmentation_SAM_2D script (basic syntax check only)."""
    # This script requires napari and user interaction, so we just check syntax
    try:
        import py_compile
        py_compile.compile("/opt/T-MIDAS/scripts/segmentation_SAM_2D.py", doraise=True)
        print("✓ segmentation_SAM_2D script syntax check passed")
        return True
    except py_compile.PyCompileError as e:
        print(f"Script syntax error: {e}")
        return False


def run_all_script_tests():
    """Run all script execution tests."""
    print("Running script execution tests...")

    tests = [
        test_remove_small_labels_script,
        test_get_basic_regionprops_script,
        test_segmentation_sam_script,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All script tests passed!")
        return True
    else:
        print("❌ Some script tests failed")
        return False


if __name__ == "__main__":
    success = run_all_script_tests()
    sys.exit(0 if success else 1)
