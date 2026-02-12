import cv2
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List


class OpticalFlowEstimator:
    def __init__(
        self,
        pyramid_levels: int = 3,
        window_size: int = 15,
        poly_n: int = 5,
        poly_sigma: float = 1.1,
        flow_scale: float = 1.0
    ):
        self.pyramid_levels = pyramid_levels
        self.window_size = window_size
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flow_scale = flow_scale
        
        self.prev_gray = None
        self.prev_points = None
    
    def _preprocess(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return gray
    
    def calculate_dense_flow(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ):
        prev_gray = self._preprocess(prev_frame)
        curr_gray = self._preprocess(curr_frame)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=self.pyramid_levels,
            winsize=self.window_size,
            iterations=3,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )
        
        return flow * self.flow_scale if flow is not None else None
    
    def calculate_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ):
        flow = self.calculate_dense_flow(prev_frame, curr_frame)
        return flow, None, None
    
    def get_average_flow(
        self,
        flow_data: np.ndarray,
    ):
        if flow_data is None or flow_data.size == 0:
            return 0.0, 0.0
        
        magnitude = np.sqrt(flow_data[..., 0]**2 + flow_data[..., 1]**2)
        mask = magnitude > 1.0
        if np.any(mask):
            dx = np.median(flow_data[mask, 0])
            dy = np.median(flow_data[mask, 1])
        else:
            dx, dy = 0.0, 0.0
        
        return float(dx), float(dy)

