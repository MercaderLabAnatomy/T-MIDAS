"""
Targeted script testing for T-MIDAS.
Tests scripts that are most likely to work with our test setup.
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


def create_test_data_for_script(script_name):
    """Create test data specific to a script."""
    test_dir = tempfile.mkdtemp()
    
    if script_name == "deep_tissue_clahe.py":
        # Create only 3D image for Deep Tissue CLAHE
        sample_3d_image = np.random.randint(0, 255, (5, 3, 10, 100, 100), dtype=np.uint8)  # T, Z, C, Y, X
        image_3d_path = os.path.join(test_dir, "sample_3d.tif")
        write_image(sample_3d_image, image_3d_path)
    else:
        # Create standard test data
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

        # Create ground truth labels for validation scripts
        gt_labels = np.zeros((100, 100), dtype=np.uint32)
        gt_labels[10:30, 10:30] = 1  # Same as labels
        gt_labels[50:70, 50:70] = 2  # Same as labels
        gt_labels_path = os.path.join(test_dir, "sample_ground_truth.tif")
        write_image(gt_labels, gt_labels_path)

        # Create ROI labels for ROI analysis scripts
        roi_labels = np.zeros((100, 100), dtype=np.uint32)
        roi_labels[5:35, 5:35] = 1  # ROI covering the first label
        roi_labels[45:75, 45:75] = 2  # ROI covering the second label
        roi_path = os.path.join(test_dir, "sample_ROI.tif")
        write_image(roi_labels, roi_path)

        # Create mask files for intersection script
        mask_labels = np.zeros((100, 100), dtype=np.uint32)
        mask_labels[10:30, 10:30] = 1
        mask_path = os.path.join(test_dir, "sample_mask.tif")
        write_image(mask_labels, mask_path)

        # Create intensity image for intersection
        intensity_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        intensity_path = os.path.join(test_dir, "sample_intensity.tif")
        write_image(intensity_image, intensity_path)

    return test_dir


def run_script_test(script_name, args, test_dir, timeout=30):
    """Run a script test with given arguments."""
    script_path = f"/opt/T-MIDAS/scripts/{script_name}"

    if not os.path.exists(script_path):
        print(f"❌ Script {script_name} not found")
        return False

    try:
        cmd = [sys.executable, script_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=test_dir)

        if result.returncode != 0:
            # Check if it's just a warning and the script actually succeeded
            if "UserWarning" in result.stderr and result.returncode == 0:
                return True
            print(f"❌ Script {script_name} failed: {result.stderr[:200]}...")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Script {script_name} timed out")
        return False
    except Exception as e:
        print(f"❌ Script {script_name} error: {e}")
        return False


def test_script_category(category_name, tests, timeout=30):
    """Test a category of scripts."""
    print(f"\n🧪 Testing {category_name} scripts:")
    print("=" * 50)

    passed = 0
    total = len(tests)

    for test_name, script_name, args in tests:
        print(f"Testing {test_name}...", end=" ")
        test_dir = create_test_data_for_script(script_name)

        try:
            success = run_script_test(script_name, args, test_dir, timeout)
            if success:
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        finally:
            shutil.rmtree(test_dir)

    print(f"\n📊 {category_name} Results: {passed}/{total} passed")
    return passed, total


def main():
    """Run comprehensive script testing."""
    print("🚀 T-MIDAS Comprehensive Script Testing")
    print("=" * 60)

    # Define test categories with scripts that are likely to work
    test_categories = [
        ("File Conversion", [
            ("Convert Instance to Semantic", "convert_instance_to_semantic.py",
             ["--input", ".", "--suffix", "_labels.tif"]),
            ("Combine Labels", "combine_labels.py",
             ["--input", ".", "--label1_tag", "_labels1.tif", "--label2_tag", "_labels2.tif", "--output", "combined.tif"]),
        ]),

        ("Segmentation", [
            ("Semantic Segmentation", "segmentation_semantic.py",
             ["--input", ".", "--threshold", "0"]),  # 0 = Otsu
            ("Instance Segmentation", "segmentation_instances.py",
             ["--input", ".", "--threshold", "0"]),  # 0 = Otsu
            ("Spot Segmentation", "segmentation_spots.py",
             ["--input", "."]),
        ]),

        ("Image Processing", [
            ("Deep Tissue CLAHE", "deep_tissue_clahe.py",
             ["--input", ".", "--kernel_size", "8", "--clip_limit", "2.0", "--nbins", "256"]),
            ("Export Labels", "export_label.py",
             ["--input", ".", "--label_id", "1"]),
        ]),

        ("ROI Analysis", [
            ("2D Instance Counting", "ROI_count_instances_2D.py",
             ["--input", ".", "--pixel_resolution", "0.5", "--roi_suffix", "_ROI.tif", "--instance_suffix", "_labels.tif"]),
            ("Get Class Regionprops", "get_class_regionprops.py",
             ["--input", ".", "--label_pattern", "_labels.tif"]),
        ]),

        ("Utility", [
            ("Random Tile Sampler", "random_tile_sampler.py",
             ["--input", ".", "--tile_diagonal", "50", "--percentage", "20"]),
            ("Intersection", "intersection.py",
             ["--input", ".", "--maskfiles", "_mask.tif", "--intersectfiles", "_intensity.tif", "--output_tag", "_intersected.tif"]),
        ]),

        ("Compression", [
            ("ZSTD Compression", "zstd_compression.py",
             ["--input_folder", ".", "--file_extension", "tif"]),
        ]),

        ("Validation", [
            ("Counts Validation", "counts_validation.py",
             ["--input", ".", "--label_pattern", "_labels.tif", "--gt_pattern", "_ground_truth.tif"]),
            ("F1 Score Validation", "segmentation_validation_f1_score.py",
             ["--input", ".", "--label_pattern", "_labels.tif", "--gt_pattern", "_ground_truth.tif"]),
        ]),
    ]

    total_passed = 0
    total_tests = 0

    for category_name, tests in test_categories:
        # Use longer timeout for validation scripts
        timeout = 60 if category_name == "Validation" else 30
        passed, total = test_script_category(category_name, tests, timeout)
        total_passed += passed
        total_tests += total

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Total Scripts Tested: {total_tests}")
    print(f"Scripts Passed: {total_passed}")
    print(".1f")

    if total_passed == total_tests:
        print("🎉 ALL SCRIPTS PASSED!")
        return True
    else:
        print("⚠️  Some scripts failed - check output above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
