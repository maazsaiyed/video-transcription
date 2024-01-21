import cv2
import sys
import traceback
from detection.face_detection import FaceDetection
from annotation.image import ImageAnnotate

if __name__ == '__main__':
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
            cv2.imshow('img', img)
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