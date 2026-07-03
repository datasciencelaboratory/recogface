import cv2
import time
import numpy as np
from typing import Optional
from recogface.camera import CameraManager
from recogface.detector import FaceDetector
from recogface.extractor import EmbeddingExtractor
from recogface.db import VectorDBClient

class PipelineController:
    """
    Orchestrates the entire face recognition pipeline in real time.
    Captures frames from camera -> Detects faces -> Extracts embeddings -> Performs vector search -> Draws visual feedback.
    """
    def __init__(self, camera_id: int = 0, threshold: float = 0.65):
        self.camera_id = camera_id
        self.threshold = threshold
        
        # Instantiate dependencies
        self.camera = CameraManager(device_id=self.camera_id)
        self.detector = FaceDetector()
        self.extractor = EmbeddingExtractor()
        self.db = VectorDBClient()

    def run(self) -> None:
        """Runs the real-time recognition loop."""
        print("\n" + "="*50)
        print("Starting Real-time Face Recognition Pipeline...")
        print("Press 'q' or 'ESC' in the output window to quit.")
        print("="*50 + "\n")
        
        try:
            self.camera.start()
        except Exception as e:
            print(f"[Error] Failed to start camera: {e}")
            print("Please ensure the camera is connected and '/dev/video0' is accessible.")
            return

        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0.0

        try:
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    print("[Warning] Failed to read frame from camera. Retrying...")
                    time.sleep(0.1)
                    continue

                # Run face detection
                faces = self.detector.detect_faces(frame)
                
                # Process each detected face
                for face in faces:
                    box = face["box"]
                    x, y, w, h = box
                    
                    # Extract face crop
                    face_crop = self.detector.crop_face(frame, box)
                    
                    name = "Desconhecido"
                    score_pct = 0.0
                    color = (0, 0, 255) # Red for unknown
                    
                    if face_crop.size > 0:
                        try:
                            # Extract embedding
                            embedding = self.extractor.extract_embedding(face_crop)
                            
                            # Search in vector DB
                            match = self.db.search_face(embedding, threshold=self.threshold)
                            if match:
                                name = match["name"]
                                score_pct = match["score"] * 100
                                color = (0, 255, 0) # Green for known
                        except Exception as e:
                            print(f"[Error] Processing face crop: {e}")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Prepare label text
                    if name != "Desconhecido":
                        label = f"{name} ({score_pct:.1f}%)"
                    else:
                        label = "Desconhecido"
                        
                    # Draw background rectangle for label
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    label_w, label_h = label_size
                    cv2.rectangle(
                        frame,
                        (x, y - label_h - 10),
                        (x + label_w + 10, y),
                        color,
                        cv2.FILLED
                    )
                    
                    # Draw text label
                    cv2.putText(
                        frame,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                # Calculate FPS
                fps_frame_count += 1
                current_time = time.time()
                elapsed = current_time - fps_start_time
                if elapsed >= 1.0:
                    fps_display = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Draw FPS indicator
                cv2.putText(
                    frame,
                    f"FPS: {fps_display:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                # Display the result
                try:
                    cv2.imshow("Recogface - Realtime Face Recognition", frame)
                except cv2.error as e:
                    # Headless mode check or X11 connection failure
                    print(
                        f"\n[X11 Error] Cannot display window: {e}.\n"
                        "Please verify your DISPLAY environment variable and host X11 permissions.\n"
                        "Make sure to run 'xhost +local:docker' on your host machine before starting.\n"
                    )
                    break

                # Break loop on 'q' or ESC (key code 27)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
        finally:
            print("\nReleasing resources and closing windows...")
            self.camera.release()
            cv2.destroyAllWindows()
            print("Pipeline shut down.")
