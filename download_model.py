import os
import sys

# Add the root directory to path to allow import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recogface.extractor import EmbeddingExtractor
except ImportError:
    # If package structure is not initialized in sys.path
    from recogface.recogface.extractor import EmbeddingExtractor

if __name__ == "__main__":
    print("Pre-downloading FaceNet 512 ONNX model...")
    # Initialize the extractor which triggers the download if model doesn't exist
    EmbeddingExtractor()
    print("Pre-downloading completed successfully.")
