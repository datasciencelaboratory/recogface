import os
import uuid
import numpy as np
from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

class VectorDBClient:
    """
    Client for managing connections and operations with Qdrant vector database.
    Creates collections, inserts facial embeddings, and queries matches.
    """
    COLLECTION_NAME = "recogface_collection"
    VECTOR_SIZE = 512

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        # Default to environment variables or local fallback
        self.host = host or os.environ.get("QDRANT_HOST", "localhost")
        self.port = port or int(os.environ.get("QDRANT_PORT", 6333))
        
        print(f"[VectorDBClient] Connecting to Qdrant at {self.host}:{self.port}...")
        self.client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Checks if the collection exists, verifies vector dimension, and creates it if not."""
        try:
            # We check if collection exists. In modern qdrant-client:
            exists = self.client.collection_exists(self.COLLECTION_NAME)
        except Exception:
            # Fallback if collection_exists method is not available in specific client version
            try:
                self.client.get_collection(self.COLLECTION_NAME)
                exists = True
            except Exception:
                exists = False
                
        if exists:
            # Verify if existing collection vector size matches our target VECTOR_SIZE (512)
            try:
                info = self.client.get_collection(self.COLLECTION_NAME)
                current_size = info.config.params.vectors.size
                if current_size != self.VECTOR_SIZE:
                    print(f"[VectorDBClient] Collection vector size mismatch (existing: {current_size}, target: {self.VECTOR_SIZE}). Re-creating collection...")
                    self.client.delete_collection(self.COLLECTION_NAME)
                    exists = False
            except Exception as e:
                print(f"[VectorDBClient] Warning during collection size validation: {e}")

        if not exists:
            print(f"[VectorDBClient] Collection '{self.COLLECTION_NAME}' not found. Creating it...")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"[VectorDBClient] Collection '{self.COLLECTION_NAME}' created.")
        else:
            print(f"[VectorDBClient] Collection '{self.COLLECTION_NAME}' is ready.")

    def add_face(self, name: str, embedding: np.ndarray) -> str:
        """
        Inserts a new face embedding vector into Qdrant.
        Args:
            name: Name of the person.
            embedding: 512-dimensional numpy array embedding vector.
        Returns:
            str: The generated UUID for the inserted point.
        """
        if len(embedding) != self.VECTOR_SIZE:
            raise ValueError(f"Embedding size must be {self.VECTOR_SIZE}, got {len(embedding)}")

        point_id = str(uuid.uuid4())
        
        # Qdrant client expects vector as list of floats
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"name": name}
                )
            ]
        )
        print(f"[VectorDBClient] Successfully registered face for '{name}' (ID: {point_id})")
        return point_id

    def search_face(self, embedding: np.ndarray, threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        Queries the database for the closest face embedding.
        Args:
            embedding: The query 128-dimensional embedding vector.
            threshold: Cosine similarity threshold (0.0 to 1.0).
        Returns:
            Dict containing {"name": str, "score": float} if similarity > threshold, else None.
        """
        if len(embedding) != self.VECTOR_SIZE:
            raise ValueError(f"Embedding size must be {self.VECTOR_SIZE}, got {len(embedding)}")

        search_result = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embedding.tolist(),
            limit=1
        )
        
        if not search_result or not search_result.points:
            return None
            
        best_match = search_result.points[0]
        score = float(best_match.score)
        
        if score >= threshold:
            name = best_match.payload.get("name", "Unknown")
            return {
                "name": name,
                "score": score,
                "id": best_match.id
            }
            
        return None
