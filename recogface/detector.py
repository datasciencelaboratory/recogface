import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Any

class FaceDetector:
    """
    Detects faces in frames using MediaPipe Face Detection.
    Optimized for fast CPU inference and supports short/long range detection.
    """
    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        # model_selection=1 is optimized for full-range faces (within 5 meters),
        # which is ideal for detecting faces from far away.
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces in the provided frame.
        Args:
            frame: Image array in BGR format.
        Returns:
            List of dictionaries representing detected faces with:
            - "box": Tuple of (x, y, width, height) in pixel coordinates
            - "confidence": Float score of detection confidence
        """
        if frame is None or frame.size == 0:
            return []

        h, w, _ = frame.shape
        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.detector.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert normalized coordinates to absolute pixels
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Boundary safety clipping
                x_clean = max(0, x)
                y_clean = max(0, y)
                w_clean = min(width, w - x_clean)
                h_clean = min(height, h - y_clean)
                
                if w_clean > 0 and h_clean > 0:
                    faces.append({
                        "box": (x_clean, y_clean, w_clean, h_clean),
                        "confidence": float(detection.score[0])
                    })
                    
        return faces

    @staticmethod
    def crop_face(frame: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.25) -> np.ndarray:
        """
        Crops a face from the frame using a bounding box, adding a margin/padding around the face
        to align it better with FaceNet's training dataset and improve embedding quality.
        Args:
            frame: Image array.
            box: Tuple of (x, y, width, height).
            margin: Percentage of padding to add around the box (0.25 = 25% padding).
        Returns:
            np.ndarray: Cropped and padded image.
        """
        x, y, w, h = box
        img_h, img_w, _ = frame.shape
        
        # Calculate padding in pixels
        pad_w = int(w * margin)
        pad_h = int(h * margin)
        
        # Calculate padded coordinates with boundary safety clipping
        x_start = max(0, x - pad_w)
        y_start = max(0, y - pad_h)
        x_end = min(img_w, x + w + pad_w)
        y_end = min(img_h, y + h + pad_h)
        
        return frame[y_start:y_end, x_start:x_end]
