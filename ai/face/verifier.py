import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceVerifier:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, image: np.ndarray):
        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected.")

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def get_document_face_embedding(self, image: np.ndarray):
        faces = self.app.get(image)

        if len(faces) == 0:
            raise ValueError("No face detected in document.")

        
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def compare(
        self,
        id_face_image: np.ndarray,
        selfie_image: np.ndarray,
        threshold: float = 0.45
    ):
        id_embedding = self.get_embedding(id_face_image)
        selfie_embedding = self.get_embedding(selfie_image)

        similarity = float(
            np.dot(id_embedding, selfie_embedding)
        )

        return {
            "similarity_score": similarity,
            "threshold": threshold,
            "match": similarity >= threshold
        }

    def compare_document_and_selfie(
        self,
        document_image: np.ndarray,
        selfie_image: np.ndarray,
        threshold: float = 0.45
    ):
        id_embedding = self.get_document_face_embedding(document_image)
        selfie_embedding = self.get_embedding(selfie_image)

        similarity = float(
            np.dot(id_embedding, selfie_embedding)
        )

        return {
            "similarity_score": similarity,
            "threshold": threshold,
            "match": similarity >= threshold
        }