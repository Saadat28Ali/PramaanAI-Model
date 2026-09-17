import cv2
import numpy as np
from pathlib import Path


class FaceVerifier:

    def __init__(self):

        
        base_dir = Path(__file__).resolve().parents[1]

        detector_model = (
            base_dir
            / "models"
            / "face"
            / "face_detection_yunet_2023mar.onnx"
        )

        recognizer_model = (
            base_dir
            / "models"
            / "face"
            / "face_recognition_sface_2021dec.onnx"
        )

        if not detector_model.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {detector_model}"
            )

        if not recognizer_model.exists():
            raise FileNotFoundError(
                f"SFace model not found: {recognizer_model}"
            )

        print("Loading YuNet face detector...")

        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model),
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        print("Loading SFace recognizer...")

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognizer_model),
            ""
        )

    def _detect_faces(self, image: np.ndarray):

        if image is None or image.size == 0:
            raise ValueError("Invalid image.")

        height, width = image.shape[:2]

        # YuNet requires the actual image size
        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        return faces

    def _get_embedding(self, image: np.ndarray, face):

        # Align and crop the detected face
        aligned_face = self.recognizer.alignCrop(
            image,
            face
        )

        # Generate face embedding
        feature = self.recognizer.feature(aligned_face)

        # Normalize embedding
        feature = feature / np.linalg.norm(feature)

        return feature

    def get_embedding(self, image: np.ndarray):

        faces = self._detect_faces(image)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected.")

        return self._get_embedding(image, faces[0])

    def get_document_face_embedding(self, image: np.ndarray):

        faces = self._detect_faces(image)

        if len(faces) == 0:
            raise ValueError("No face detected in document.")

        # Select largest detected face
        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        return self._get_embedding(image, face)

    def compare(
        self,
        id_face_image: np.ndarray,
        selfie_image: np.ndarray,
        threshold: float = 0.363
    ):

        id_embedding = self.get_embedding(id_face_image)
        selfie_embedding = self.get_embedding(selfie_image)

        similarity = float(
            np.dot(
                id_embedding.flatten(),
                selfie_embedding.flatten()
            )
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
        threshold: float = 0.363
    ):

        id_embedding = self.get_document_face_embedding(
            document_image
        )

        selfie_embedding = self.get_embedding(
            selfie_image
        )

        similarity = float(
            np.dot(
                id_embedding.flatten(),
                selfie_embedding.flatten()
            )
        )

        return {
            "similarity_score": similarity,
            "threshold": threshold,
            "match": similarity >= threshold
        }