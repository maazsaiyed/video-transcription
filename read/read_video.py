import os
import cv2
import sys
import traceback
from detection.face_detection import FaceDetection
from annotation.image import ImageAnnotate


class ReadVideo:
    def __init__(self):
        pass

    def from_webcam(self) -> None:
        """Capture live video from webcam and processes for face detection."""
        cap = cv2.VideoCapture(1)

        face_det = FaceDetection()
        image_annotation = ImageAnnotate()

        try:
            while True:
                # read an image
                _, img = cap.read()
                # detect faces from image
                location = face_det.detect(img)
                # draw bounding box
                img = image_annotation.annotate(img, location)
                # display picture
                cv2.imshow("img", img)
                # Wait for Esc k`ey to stop
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        except Exception as err:
            traceback.print_stack(sys.stderr, err)
        finally:
            if cap:
                cap.release()
                cv2.destroyAllWindows()

    def from_path(self, file_path: str, fps: int = 5) -> None:
        """Reads video file from a file path and processes for face detection."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Video file not available")

        cap = cv2.VideoCapture(file_path)
        frame_count = 0

        face_det = FaceDetection()
        image_annotation = ImageAnnotate()

        try:
            while cap.isOpened():
                # read an image
                ret, img = cap.read()
                if not ret:
                    break
                frame_count = frame_count + 1
                if frame_count % fps == 0:
                    # detect faces from image
                    location = face_det.detect(img)
                    # draw bounding box
                    img = image_annotation.annotate(img, location)
                    # display picture
                    cv2.imshow("img", img)
                    # Wait for Esc k`ey to stop
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
        except Exception as err:
            traceback.print_stack(sys.stderr, err)
        finally:
            if cap:
                cap.release()
                cv2.destroyAllWindows()

