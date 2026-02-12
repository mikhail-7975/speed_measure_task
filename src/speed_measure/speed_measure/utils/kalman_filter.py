import numpy as np

class KalmanFilter:
    """
    Simple 1D Kalman filter for altitude smoothing (float32).
    Assumes constant position model: altitude doesn't change between measurements.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        """
        Initialize filter.
        
        Args:
            process_noise: Process noise variance (Q) - how much we trust our motion model
            measurement_noise: Measurement noise variance (R) - how much we trust sensor readings
        """
        # State: [altitude]
        self.x = np.float32(0.0)  # Current altitude estimate
        self.P = np.float32(100.0)  # Error covariance
        
        # Noise parameters (converted to float32)
        self.Q = np.float32(process_noise)   # Process noise
        self.R = np.float32(measurement_noise)  # Measurement noise
        
    def update(self, measurement: float) -> float:
        """
        Update filter with new altitude measurement.
        
        Args:
            measurement: Raw altitude reading (will be cast to float32)
            
        Returns:
            Filtered altitude estimate (float32)
        """
        z = np.float32(measurement)
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P = (1 - K) * self.P
        return float(self.x)
    
    def reset(self, initial_altitude: float = 0.0) -> None:
        """Reset filter state."""
        self.x = np.float32(initial_altitude)
        self.P = np.float32(100.0)