"""
Test to demonstrate the resize issue with label images
"""
import numpy as np
from skimage.transform import resize

# Create a simple label image
labels = np.zeros((100, 100), dtype=np.uint16)
labels[10:40, 10:40] = 1
labels[50:80, 50:80] = 2

print("Original labels:")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label counts: {[(lab, np.sum(labels == lab)) for lab in np.unique(labels) if lab != 0]}")

# Resize with anti-aliasing (current behavior)
resized_aa = resize(labels, (90, 90), anti_aliasing=True, preserve_range=True)
resized_aa = np.round(resized_aa).astype(np.uint16)

print("\nResized WITH anti-aliasing:")
print(f"Unique labels: {np.unique(resized_aa)}")
unique_nonzero = np.unique(resized_aa[resized_aa != 0])
print(f"Unique non-zero labels: {unique_nonzero}")
print(f"Number of spurious labels: {len(unique_nonzero) - 2}")  # Should be 2 labels (1 and 2)

# Resize without anti-aliasing (correct behavior for labels)
resized_no_aa = resize(labels, (90, 90), anti_aliasing=False, preserve_range=True, order=0)
resized_no_aa = np.round(resized_no_aa).astype(np.uint16)

print("\nResized WITHOUT anti-aliasing (order=0, nearest neighbor):")
print(f"Unique labels: {np.unique(resized_no_aa)}")
unique_nonzero = np.unique(resized_no_aa[resized_no_aa != 0])
print(f"Unique non-zero labels: {unique_nonzero}")
print(f"Number of spurious labels: {len(unique_nonzero) - 2}")

print("\n" + "="*60)
print("CONCLUSION:")
if len(np.unique(resized_aa[resized_aa != 0])) > 2:
    print("❌ Anti-aliasing creates spurious labels!")
    print("   This will cause inflated counts in colocalization analysis.")
else:
    print("✓ No issues detected")
