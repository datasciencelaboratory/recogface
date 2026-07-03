import cv2
import numpy as np
from typing import Optional

class CameraManager:
    """
    Manages OpenCV video capture from a USB webcam.
    Provides context manager support to ensure proper release of resources.
    """
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Initializes the video capture device."""
        if self.cap is not None and self.cap.isOpened():
            return
        
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video device {self.device_id}")
        
        # Configure reasonable defaults for real-time capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Captures a single frame from the camera.
        Returns:
            np.ndarray: The captured frame in BGR format, or None if reading failed.
        """
        if self.cap is None or not self.cap.isOpened():
            self.start()
            
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def release(self) -> None:
        """Releases the camera resource."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
