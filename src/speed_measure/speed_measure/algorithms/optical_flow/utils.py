import cv2
import numpy as np


def visualize_dense_flow(
        flow: np.ndarray,
        magnitude_threshold: float = 1.0
    ) -> np.ndarray:
        if flow is None:
            raise ValueError("Flow input is None")
        
        # Compute magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Mask small motions
        mask = magnitude > magnitude_threshold
        
        # HSV visualization
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255                      # Saturation max
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value = magnitude
        
        # Apply mask to hide noise
        hsv[~mask] = 0
        
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr