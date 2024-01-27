import cv2
import numpy as np
from typing import Sequence
from model.rectangle_model import Rectangle
from model.point_model import Point


class ImageAnnotate:
    @classmethod
    def rectangle(cls, image: np.ndarray, location: Rectangle) -> np.ndarray:
        """Annotate images using the locations"""

        annotated_image: np.ndarray = image.copy()

        cv2.rectangle(
            img=annotated_image,
            pt1=location.get_top_left.get_tuple(),
            pt2=location.get_bottom_right.get_tuple(),
            color=(0, 0, 255),
            thickness=2
        )

        return annotated_image

    @classmethod
    def text(cls, image: np.ndarray, text: str, location: Point) -> np.ndarray:
        """Writes text on an image"""

        annotated_image: np.ndarray = image.copy()

        cv2.putText(
            img=annotated_image,
            text=text,
            org=location.get_tuple(),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2
        )

        return annotated_image
