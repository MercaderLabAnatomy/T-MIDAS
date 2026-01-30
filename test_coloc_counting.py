"""
Test script for ROI_colocalization_count_multicolor.py with synthetic data
Creates test images with known counts to verify the counting logic.
"""

import numpy as np
import os
import tempfile
import shutil
from skimage import io
import sys

def create_synthetic_test_data():
    """
    Create synthetic test images with known colocalization patterns.
    
    Returns:
        tuple: (temp_dir, expected_counts_dict)
    """
    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp(prefix='coloc_test_')
    
    # Create channel directories
    ch1_dir = os.path.join(temp_dir, 'Channel1')
    ch2_dir = os.path.join(temp_dir, 'Channel2')
    ch3_dir = os.path.join(temp_dir, 'Channel3')
    
    os.makedirs(ch1_dir)
    os.makedirs(ch2_dir)
    os.makedirs(ch3_dir)
    
    print(f"Created test directory: {temp_dir}")
    
    # Test Case 1: Simple non-overlapping labels
    print("\n=== Test Case 1: Simple non-overlapping labels ===")
    c1_img1 = np.zeros((100, 100), dtype=np.uint16)
    c1_img1[10:30, 10:30] = 1  # ROI 1
    c1_img1[40:60, 40:60] = 2  # ROI 2
    c1_img1[70:90, 10:30] = 3  # ROI 3
    
    c2_img1 = np.zeros((100, 100), dtype=np.uint16)
    c2_img1[15:25, 15:25] = 10  # 1 object in ROI 1
    c2_img1[15:25, 16:26] = 11  # 1 object in ROI 1 (overlapping with 10)
    c2_img1[45:55, 45:55] = 20  # 1 object in ROI 2
    c2_img1[46:56, 46:56] = 21  # 1 object in ROI 2
    c2_img1[47:57, 47:57] = 22  # 1 object in ROI 2
    # ROI 3 has no objects
    
    c3_img1 = np.zeros((100, 100), dtype=np.uint16)
    c3_img1[16:24, 16:24] = 100  # 1 object overlapping with c2 label 10 and 11 in ROI 1
    c3_img1[46:54, 46:54] = 200  # 1 object overlapping with c2 labels 20,21,22 in ROI 2
    c3_img1[75:85, 15:25] = 300  # 1 object in ROI 3 (no c2 overlap)
    
    # Save images
    io.imsave(os.path.join(ch1_dir, 'test1_labels.tif'), c1_img1)
    io.imsave(os.path.join(ch2_dir, 'test1_labels.tif'), c2_img1)
    io.imsave(os.path.join(ch3_dir, 'test1_labels.tif'), c3_img1)
    
    # Expected counts for test1
    expected_test1 = {
        'ROI_1': {'c2_count': 2, 'c3_in_c2_count': 1, 'c3_not_in_c2_count': 0},
        'ROI_2': {'c2_count': 3, 'c3_in_c2_count': 1, 'c3_not_in_c2_count': 0},
        'ROI_3': {'c2_count': 0, 'c3_in_c2_count': 0, 'c3_not_in_c2_count': 1}
    }
    
    print("Test 1 Ground Truth:")
    print(f"  ROI 1: {expected_test1['ROI_1']['c2_count']} C2 objects, "
          f"{expected_test1['ROI_1']['c3_in_c2_count']} C3 in C2, "
          f"{expected_test1['ROI_1']['c3_not_in_c2_count']} C3 not in C2")
    print(f"  ROI 2: {expected_test1['ROI_2']['c2_count']} C2 objects, "
          f"{expected_test1['ROI_2']['c3_in_c2_count']} C3 in C2, "
          f"{expected_test1['ROI_2']['c3_not_in_c2_count']} C3 not in C2")
    print(f"  ROI 3: {expected_test1['ROI_3']['c2_count']} C2 objects, "
          f"{expected_test1['ROI_3']['c3_in_c2_count']} C3 in C2, "
          f"{expected_test1['ROI_3']['c3_not_in_c2_count']} C3 not in C2")
    
    # Test Case 2: Complex overlapping patterns
    print("\n=== Test Case 2: Complex overlapping patterns ===")
    c1_img2 = np.zeros((100, 100), dtype=np.uint16)
    c1_img2[10:90, 10:90] = 1  # Single large ROI
    
    c2_img2 = np.zeros((100, 100), dtype=np.uint16)
    # Create 5 distinct objects
    c2_img2[15:25, 15:25] = 1
    c2_img2[30:40, 30:40] = 2
    c2_img2[45:55, 45:55] = 3
    c2_img2[60:70, 60:70] = 4
    c2_img2[75:85, 75:85] = 5
    
    c3_img2 = np.zeros((100, 100), dtype=np.uint16)
    # 2 objects overlapping with c2, 1 object not overlapping
    c3_img2[17:23, 17:23] = 10  # overlaps with c2 object 1
    c3_img2[32:38, 32:38] = 11  # overlaps with c2 object 2
    c3_img2[50:58, 20:28] = 12  # does NOT overlap with any c2 object
    
    io.imsave(os.path.join(ch1_dir, 'test2_labels.tif'), c1_img2)
    io.imsave(os.path.join(ch2_dir, 'test2_labels.tif'), c2_img2)
    io.imsave(os.path.join(ch3_dir, 'test2_labels.tif'), c3_img2)
    
    expected_test2 = {
        'ROI_1': {'c2_count': 5, 'c3_in_c2_count': 2, 'c3_not_in_c2_count': 1}
    }
    
    print("Test 2 Ground Truth:")
    print(f"  ROI 1: {expected_test2['ROI_1']['c2_count']} C2 objects, "
          f"{expected_test2['ROI_1']['c3_in_c2_count']} C3 in C2, "
          f"{expected_test2['ROI_1']['c3_not_in_c2_count']} C3 not in C2")
    
    expected_counts = {
        'test1': expected_test1,
        'test2': expected_test2
    }
    
    return temp_dir, expected_counts

def verify_unique_counting():
    """
    Test the count_unique_nonzero function directly with known arrays.
    """
    print("\n=== Testing count_unique_nonzero function directly ===")
    
    # Import the function
    sys.path.insert(0, '/opt/T-MIDAS/scripts')
    from ROI_colocalization_count_multicolor import count_unique_nonzero
    
    # Test case 1: Array with labels 1, 2, 3
    array1 = np.array([0, 1, 1, 2, 2, 2, 3, 3, 0])
    mask1 = np.array([True, True, True, True, True, True, True, True, True])
    result1 = count_unique_nonzero(array1, mask1)
    expected1 = 3  # Should be 3 (labels 1, 2, 3)
    print(f"Test 1: Expected {expected1}, Got {result1} - {'PASS' if result1 == expected1 else 'FAIL'}")
    
    # Test case 2: Array with only zeros
    array2 = np.zeros(10, dtype=np.uint16)
    mask2 = np.ones(10, dtype=bool)
    result2 = count_unique_nonzero(array2, mask2)
    expected2 = 0  # Should be 0
    print(f"Test 2: Expected {expected2}, Got {result2} - {'PASS' if result2 == expected2 else 'FAIL'}")
    
    # Test case 3: Array with labels and mask filtering some out
    array3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mask3 = np.array([True, True, True, False, False, False, False, False, False, False])
    result3 = count_unique_nonzero(array3, mask3)
    expected3 = 3  # Should be 3 (labels 1, 2, 3)
    print(f"Test 3: Expected {expected3}, Got {result3} - {'PASS' if result3 == expected3 else 'FAIL'}")
    
    # Test case 4: 2D array with overlapping labels
    array4 = np.array([[1, 1, 2, 2],
                       [1, 1, 2, 2],
                       [3, 3, 0, 0],
                       [3, 3, 0, 0]])
    mask4 = np.array([[True, True, True, True],
                      [True, True, True, True],
                      [True, True, False, False],
                      [True, True, False, False]])
    result4 = count_unique_nonzero(array4, mask4)
    expected4 = 3  # Should be 3 (labels 1, 2, 3)
    print(f"Test 4: Expected {expected4}, Got {result4} - {'PASS' if result4 == expected4 else 'FAIL'}")

def parse_csv_results(csv_file):
    """Parse the results CSV and extract counts."""
    import csv
    results = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename']
            roi_id = int(row['Channel1_label_id'])
            c2_count = int(row['Channel2_in_Channel1_count'])
            c3_in_c2_count = int(row['Channel3_in_Channel2_in_Channel1_count'])
            c3_not_in_c2_count = int(row['Channel3_not_in_Channel2_but_in_Channel1_count'])
            
            test_name = filename.replace('_labels.tif', '')
            if test_name not in results:
                results[test_name] = {}
            
            results[test_name][f'ROI_{roi_id}'] = {
                'c2_count': c2_count,
                'c3_in_c2_count': c3_in_c2_count,
                'c3_not_in_c2_count': c3_not_in_c2_count
            }
    
    return results

def compare_results(expected, actual):
    """Compare expected and actual results."""
    print("\n=== Comparison Results ===")
    all_passed = True
    
    for test_name, test_expected in expected.items():
        print(f"\n{test_name}:")
        test_actual = actual.get(test_name, {})
        
        for roi_name, roi_expected in test_expected.items():
            roi_actual = test_actual.get(roi_name, {})
            
            c2_match = roi_expected['c2_count'] == roi_actual.get('c2_count', -1)
            c3_in_match = roi_expected['c3_in_c2_count'] == roi_actual.get('c3_in_c2_count', -1)
            c3_not_match = roi_expected['c3_not_in_c2_count'] == roi_actual.get('c3_not_in_c2_count', -1)
            
            status = 'PASS' if (c2_match and c3_in_match and c3_not_match) else 'FAIL'
            if status == 'FAIL':
                all_passed = False
            
            print(f"  {roi_name}: {status}")
            print(f"    C2 count: Expected {roi_expected['c2_count']}, Got {roi_actual.get('c2_count', 'N/A')}")
            print(f"    C3 in C2: Expected {roi_expected['c3_in_c2_count']}, Got {roi_actual.get('c3_in_c2_count', 'N/A')}")
            print(f"    C3 not in C2: Expected {roi_expected['c3_not_in_c2_count']}, Got {roi_actual.get('c3_not_in_c2_count', 'N/A')}")
    
    return all_passed

def main():
    print("="*60)
    print("Testing ROI Colocalization Counting Logic")
    print("="*60)
    
    # First, test the counting function directly
    verify_unique_counting()
    
    # Create synthetic test data
    temp_dir, expected_counts = create_synthetic_test_data()
    
    try:
        # Run the colocalization script
        print("\n=== Running colocalization analysis ===")
        cmd = f"""python /opt/T-MIDAS/scripts/ROI_colocalization_count_multicolor.py \
            --input {temp_dir} \
            --channels Channel1 Channel2 Channel3 \
            --label_patterns "*_labels.tif" "*_labels.tif" "*_labels.tif" \
            --channel3_is_labels y \
            --get_sizes n \
            --no_resize"""
        
        print(f"Command: {cmd}")
        os.system(cmd)
        
        # Parse results
        csv_file = os.path.join(temp_dir, 'Channel1_Channel2_Channel3_colocalization.csv')
        if os.path.exists(csv_file):
            actual_results = parse_csv_results(csv_file)
            
            # Compare results
            all_passed = compare_results(expected_counts, actual_results)
            
            print("\n" + "="*60)
            if all_passed:
                print("ALL TESTS PASSED ✓")
            else:
                print("SOME TESTS FAILED ✗")
            print("="*60)
        else:
            print(f"ERROR: Results CSV not found at {csv_file}")
    
    finally:
        # Cleanup
        response = input(f"\nKeep test directory {temp_dir}? (y/n): ")
        if response.lower() != 'y':
            shutil.rmtree(temp_dir)
            print("Test directory removed.")
        else:
            print(f"Test directory preserved at: {temp_dir}")

if __name__ == "__main__":
    main()
