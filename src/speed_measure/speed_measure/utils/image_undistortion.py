import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple


class CameraUndistorter:
    """
    Camera undistortion utility with callable interface.
    
    Precomputes undistortion maps for optimal performance.
    Supports both file paths and numpy arrays as input.
    """
    
    def __init__(
        self,
        intrinsic_matrix: Union[list, np.ndarray],
        distortion_coeffs: Union[list, np.ndarray],
        image_size: Optional[Tuple[int, int]] = None,
        optimize_fov: float = 0.0
    ):
        """
        Initialize undistorter with camera parameters.
        
        Args:
            intrinsic_matrix: 9-element list/array [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            distortion_coeffs: 5-element list/array [k1, k2, p1, p2, k3]
            image_size: (width, height) of images to process. Required if not set later.
            optimize_fov: Optional scaling factor (0.0 to 1.0) to preserve FOV after undistortion.
                          Values >0 reduce black borders but may crop image edges.
        """
        # Parse intrinsic matrix
        self.K = np.array(intrinsic_matrix, dtype=np.float32).reshape(3, 3)
        
        # Parse distortion coefficients (ensure 5 elements)
        self.dist = np.array(distortion_coeffs, dtype=np.float32).flatten()
        if self.dist.size != 5:
            raise Exception("dist.size != 5")
        
        self.image_size = image_size
        self.optimize_fov = max(0.0, min(1.0, float(optimize_fov)))
        self.map1 = None
        self.map2 = None
        
        # Precompute maps if image size is known
        if image_size is not None:
            self._precompute_maps(image_size)
    
    def _precompute_maps(self, image_size: Tuple[int, int]) -> None:
        """Precompute undistortion maps for fast remapping."""
        width, height = image_size
        
        # Optional: Adjust camera matrix to preserve FOV
        if self.optimize_fov > 0:
            new_K, roi = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (width, height), 
                alpha=self.optimize_fov, newImgSize=(width, height)
            )
            self.roi = roi
        else:
            new_K = self.K
            self.roi = (0, 0, width, height)
        
        # Precompute maps
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, new_K, (width, height), cv2.CV_16SC2
        )
        self.image_size = (width, height)
        self.new_K = new_K
    
    def __call__(
        self, 
        image: Union[str, Path, np.ndarray], 
        crop_valid_region: bool = False
    ) -> np.ndarray:
        """
        Undistort an image. Callable interface for convenient use.
        
        Args:
            image: Path to image file or numpy array (BGR format)
            image_size: (width, height) if not set during initialization
            crop_valid_region: If True, crop to region without black borders
        
        Returns:
            Undistorted image as numpy array (BGR format)
        """
        if isinstance(image, np.ndarray):
            img = image.copy()  # Avoid modifying original
        else:
            raise TypeError("image must be numpy array")
        
        # Set image size if not already known
        if self.image_size is None:
            h, w = img.shape[:2]
            self._precompute_maps((w, h))
        
        # Apply undistortion using precomputed maps (fastest method)
        undistorted = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
        
        # Optional cropping to valid region
        if crop_valid_region and self.roi is not None:
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
