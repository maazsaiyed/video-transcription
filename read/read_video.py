import os
import cv2
import sys
import traceback
from detection.face_detection import FaceDetection
from annotation.image import ImageAnnotate


class ReadVideo:

    def from_webcam(self) -> None:
        """Capture live video from webcam and processes for face detection."""
        cap = cv2.VideoCapture(1)

        face_det = FaceDetection()

        try:
            while True:
                # read an image
                _, img = cap.read()
                # detect faces from image
                location = face_det.detect(img)
                # draw bounding box
                img = ImageAnnotate.rectangle(img, location)
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

    def from_path(self, file_path: str, fps: int = 30) -> None:
        """Reads video file from a file path and processes for face detection."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError("Video file not available")

        cap = cv2.VideoCapture(file_path)
        frame_count = 0

        face_det = FaceDetection()

        try:
            while cap.isOpened():
                # read an image
                ret, img = cap.read()
                if not ret:
                    break
                frame_count = frame_count + 1
                if frame_count % fps == 0:
                    # detect faces from image
                    face_locations = face_det.detect(img)

                    for face_loc in face_locations:
                        # draw bounding box
                        img = ImageAnnotate.rectangle(img, face_loc)

                        # face recognition for each detected face
                        crop_image = img[
                            face_loc.get_top_left.y : face_loc.get_bottom_right.y,
                            face_loc.get_top_left.x : face_loc.get_bottom_right.x
                        ]
                        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                        recognized_face = face_det.recognize(crop_image)

                        if recognized_face:
                            # write text
                            img = ImageAnnotate.text(img, text=str(recognized_face["id"]), location=face_loc.get_top_left)

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

