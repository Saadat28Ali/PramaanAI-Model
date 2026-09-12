import cv2

from ai.face.verifier import FaceVerifier


def main():
    print("Loading face verification model...")

    verifier = FaceVerifier()

    print("Model loaded successfully!")

    id_image = cv2.imread("ai/face/test.jpg")
    selfie_image = cv2.imread("ai/face/image.png")

    if id_image is None or selfie_image is None:
        print("One or both images not found!")
        return

    result = verifier.compare(
        id_face_image=id_image,
        selfie_image=selfie_image
    )

    print("\nFace Verification Result")
    print("------------------------")
    print("Similarity Score:", result["similarity_score"])
    print("Threshold:", result["threshold"])
    print("Match:", result["match"])


if __name__ == "__main__":
    main()