import argparse
import sys
import os
import cv2
from typing import List

from recogface.detector import FaceDetector
from recogface.extractor import EmbeddingExtractor
from recogface.db import VectorDBClient
from recogface.pipeline import PipelineController

def register_face(name: str, image_path: str) -> None:
    """
    Detects face in image, extracts embedding, and saves to database.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading image: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Failed to load image: {image_path}. Check if file is a valid image.", file=sys.stderr)
        sys.exit(1)

    # Initialize detector, extractor and db client
    try:
        detector = FaceDetector()
        extractor = EmbeddingExtractor()
        db = VectorDBClient()
    except Exception as e:
        print(f"[Error] Failed to initialize system components: {e}", file=sys.stderr)
        sys.exit(1)

    print("Detecting face in image...")
    faces = detector.detect_faces(img)

    if not faces:
        print("[Error] No face detected in the image. Registration aborted.", file=sys.stderr)
        sys.exit(1)

    if len(faces) > 1:
        print(f"[Warning] {len(faces)} faces detected. Selecting the largest face for registration...")
        # Sort by area (width * height) descending
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)

    target_face = faces[0]
    box = target_face["box"]
    print(f"Face detected with confidence: {target_face['confidence']:.2%}")

    face_crop = detector.crop_face(img, box)
    if face_crop.size == 0:
        print("[Error] Extracted face crop is empty. Registration aborted.", file=sys.stderr)
        sys.exit(1)

    try:
        print("Extracting embedding vector...")
        embedding = extractor.extract_embedding(face_crop)
        
        print(f"Saving vector to Qdrant collection...")
        db.add_face(name, embedding)
        print(f"Success: '{name}' has been successfully registered in recogface.")
    except Exception as e:
        print(f"[Error] Failed during embedding extraction or DB insertion: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="recogface - Realtime Face Recognition System with Qdrant, MediaPipe, and MobileFaceNet."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'add' command
    parser_add = subparsers.add_parser("add", help="Register a new face in the vector database")
    parser_add.add_argument("--name", required=True, type=str, help="Name of the person to register")
    parser_add.add_argument("--image", required=True, type=str, help="Path to the JPEG/PNG face image")

    # 'run' command
    parser_run = subparsers.add_parser("run", help="Start the real-time webcam face recognition pipeline")
    parser_run.add_argument("--camera", default=0, type=int, help="Webcam device ID (default: 0)")
    parser_run.add_argument("--threshold", default=0.65, type=float, help="Cosine similarity search threshold (default: 0.65)")

    args = parser.parse_args()

    if args.command == "add":
        register_face(name=args.name, image_path=args.image)
    elif args.command == "run":
        controller = PipelineController(camera_id=args.camera, threshold=args.threshold)
        controller.run()
    else:
        # If no arguments or invalid command, print help
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
