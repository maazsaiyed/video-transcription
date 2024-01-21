import cv2
import numpy as np
from typing import Sequence
from model.rectangle_model import Rectangle


class FaceDetection:

    _FACE_HAARCASCADE_PATH: str = "resources/haarcascade/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FaceDetection._FACE_HAARCASCADE_PATH)

    def detect(self, image: np.ndarray) -> Sequence[Rectangle]:
        """Detects human faces from an image returns position"""
        # bgr image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces using haarcascade on grayscale image
        faces: Sequence[Rectangle] = list(map(Rectangle.from_haarcascade, self.face_cascade.detectMultiScale(gray_image)))

        return faces
