import cv2
import numpy as np
from typing import Sequence
from model.rectangle_model import Rectangle

class ImageAnnotate:

    def annotate(self, image: np.ndarray, location: Sequence[Rectangle]) -> np.ndarray:
        """Annotate images using the locations"""

        annotated_image: np.ndarray = image.copy()

        for rect in location:
            cv2.rectangle(
                img=annotated_image,
                pt1=rect.get_top_left.get_tuple(),
                pt2=rect.get_bottom_right.get_tuple(),
                color=(255, 0, 0),
                thickness=2
            )

        return annotated_image
