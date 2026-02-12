import sys
from pathlib import Path
from enum import Enum
from typing import Tuple, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from speed_measure.speed_measure.algorithms.optical_flow.optical_flow_estimator import OpticalFlowEstimator


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    estimator = OpticalFlowEstimator()
    ret, prev_frame = cap.read()
    
    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        flow = estimator.calculate_dense_flow(prev_frame, curr_frame)
        if flow is not None:
            vis = estimator.visualize_dense_flow(flow)
            cv2.imshow("Dense Flow", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_frame = curr_frame
    
    cap.release()
    cv2.destroyAllWindows()