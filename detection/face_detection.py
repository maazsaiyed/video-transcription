import cv2
import face_recognition
import numpy as np
from typing import Sequence, List
from model.rectangle_model import Rectangle
from typing import Optional


class FaceDetection:

    _FACE_HAARCASCADE_PATH: str = "resources/haarcascade/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FaceDetection._FACE_HAARCASCADE_PATH)

        self.saved_faces: List[dict] = []

    def detect(self, image: np.ndarray) -> Sequence[Rectangle]:
        """Detects human faces from an image returns position"""
        # bgr image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces using haarcascade on grayscale image
        faces: Sequence[Rectangle] = list(map(Rectangle.from_haarcascade, self.face_cascade.detectMultiScale(gray_image)))

        return faces

    def recognize(self, face_image: np.ndarray) -> Optional[dict]:
        """Matches with already recognized faces from the list. If not found adds it to the list"""
        face_encoding = face_recognition.face_encodings(face_image)
        if face_encoding:
            face_encoding = face_encoding[0]

            if self.saved_faces:
                compare_result = face_recognition.compare_faces(list(map(lambda x: x["face_encoding"], self.saved_faces)), face_encoding)
                distance_result = face_recognition.face_distance(list(map(lambda x: x["face_encoding"], self.saved_faces)), face_encoding)
                if compare_result[np.argmin(distance_result)]:
                    return self.saved_faces[np.argmin(distance_result)]
                else:
                    self.saved_faces.append({
                        "image": face_image,
                        "face_encoding": face_encoding,
                        "id": len(self.saved_faces) + 1
                    })
            else:
                self.saved_faces.append({
                    "image": face_image,
                    "face_encoding": face_encoding,
                    "id": len(self.saved_faces) + 1
                })