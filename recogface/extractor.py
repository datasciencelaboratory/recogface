import cv2
import numpy as np
import os
import urllib.request
from typing import Optional

class EmbeddingExtractor:
    """
    Extracts high-quality face embeddings from cropped face images.
    Uses a pre-trained FaceNet 512-dimensional ONNX model running on OpenCV DNN (CPU-optimized).
    """
    MODEL_URL = "https://huggingface.co/haikalmumtaz/facenet-onnx/resolve/main/facenet.onnx"
    
    def __init__(self, model_dir: str = "models"):
        # Put models folder inside the app workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(base_dir, model_dir)
        self.model_path = os.path.join(self.model_dir, "facenet.onnx")
        
        self.net = None
        self._load_model()

    def _load_model(self) -> None:
        """Downloads the ONNX model if not present, and loads it into OpenCV DNN."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_dir, exist_ok=True)
            print(f"[EmbeddingExtractor] Downloading FaceNet 512 model from {self.MODEL_URL}...")
            try:
                # Custom User-Agent to avoid getting blocked by HuggingFace CDN blocks
                req = urllib.request.Request(
                    self.MODEL_URL,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(self.model_path, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)
                print(f"[EmbeddingExtractor] Model saved to {self.model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download the FaceNet model: {e}. "
                    "Ensure you have an internet connection."
                )

        # Load the ONNX model using OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Configure model to run on CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def extract_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Extracts a 512-dimensional L2-normalized embedding vector from a face crop.
        Args:
            face_crop: The cropped face image in BGR format.
        Returns:
            np.ndarray: A 512-dimensional normalized float vector.
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Input face crop is empty or invalid.")

        # Preprocessing for FaceNet (InceptionResnetV1):
        # 1. Resize to 160x160
        # 2. Swap BGR to RGB (swapRB=True)
        # 3. Normalize values using (x - 127.5) / 128.0
        #    Therefore, scalefactor = 1.0 / 128.0, and mean = 127.5
        
        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0 / 128.0,
            size=(160, 160),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        
        self.net.setInput(blob)
        embedding = self.net.forward()
        
        # Squeeze output to 1D array
        embedding = embedding.flatten()
        
        # Perform L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
