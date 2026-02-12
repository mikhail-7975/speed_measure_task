# tests/test_something.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from speed_measure.speed_measure.utils.image_undistortion import CameraUndistorter

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Camera parameters
    intrinsic_matrix = [
        537.0, 0.0, 476.30553261,
        0.0, 532.9, 272.7,
        0.0, 0.0, 1.0
    ]
    
    distortion_coeffs = [
        -0.08050122007374724,
        0.06195116455291804,
        -9.05187506022231e-05,
        0.0023246746346194213,
        0.0
    ]
    
    # Initialize undistorter (image size auto-detected on first call)
    undistorter = CameraUndistorter(
        intrinsic_matrix=intrinsic_matrix,
        distortion_coeffs=distortion_coeffs,
        optimize_fov=0.8  # Preserve more of the field of view
    )
    
    # Method 2: Process numpy array directly
    try:
        img_array = cv2.imread("/tmp/speed_measure_tmp/dataset/2026-02-10 20:52:00.429602.png")
        if img_array is not None:
            result = undistorter(img_array)  # <-- __call__ with array
            print(f"✓ Array processed: {result.shape}")
            cv2.imwrite("/tmp/speed_measure_tmp/__tmp_data/undistorted.png", result)
    except Exception as e:
        print(f"Error processing array: {e}")
    
    
