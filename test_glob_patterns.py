"""
Test glob patterns with varying label suffixes
"""
import os
import glob
import tempfile
import shutil

def test_glob_patterns():
    """Test different glob patterns for matching files with varying suffixes."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix='glob_test_')
    
    try:
        # Create test files with varying label suffixes
        test_files = [
            'image_A_labels1.tif',
            'image_A_labels5.tif',
            'image_A_labels23.tif',
            'image_B_labels2.tif',
            'image_B_labels10.tif',
            'image_B_labels456.tif',
            'image_C_labels.tif',  # No number
            'image_D_labelsX.tif',  # Letter instead of number
        ]
        
        for filename in test_files:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write('test')
        
        print(f"Created test directory: {temp_dir}")
        print(f"Test files: {test_files}\n")
        
        # Test different patterns
        patterns = [
            ("*_labels*.tif", "Match all files with '_labels' anywhere"),
            ("*_labels[0-9]*.tif", "Match files with '_labels' followed by at least one digit"),
            ("*_labels?.tif", "Match files with '_labels' followed by exactly one character"),
            ("*_labels[0-9].tif", "Match files with '_labels' followed by exactly one digit"),
            ("*_labels[0-9][0-9]*.tif", "Match files with '_labels' followed by at least two digits"),
        ]
        
        for pattern, description in patterns:
            full_pattern = os.path.join(temp_dir, pattern)
            matched = sorted([os.path.basename(f) for f in glob.glob(full_pattern)])
            
            print(f"Pattern: {pattern}")
            print(f"Description: {description}")
            print(f"Matched {len(matched)} files:")
            for f in matched:
                print(f"  - {f}")
            print()
        
        # Test for your specific use case
        print("="*60)
        print("RECOMMENDED PATTERNS FOR YOUR USE CASE:")
        print("="*60)
        print("\nIf you have files like: _labels1, _labels23, _labels5, etc.")
        print("Use pattern: \"*_labels[0-9]*.tif\"")
        print("This matches files with '_labels' followed by one or more digits\n")
        
        pattern = "*_labels[0-9]*.tif"
        full_pattern = os.path.join(temp_dir, pattern)
        matched = sorted([os.path.basename(f) for f in glob.glob(full_pattern)])
        print(f"Example matches with pattern '{pattern}':")
        for f in matched:
            print(f"  ✓ {f}")
        
        # Show what doesn't match
        all_files = set(test_files)
        matched_set = set(matched)
        not_matched = all_files - matched_set
        if not_matched:
            print(f"\nFiles NOT matched:")
            for f in sorted(not_matched):
                print(f"  ✗ {f}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nTest directory cleaned up.")

if __name__ == "__main__":
    test_glob_patterns()
