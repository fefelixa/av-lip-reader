"""
OpenCV documentation:
https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html
https://docs.opencv.org/4.11.0/d1/de5/classcv_1_1CascadeClassifier.html
models:
https://github.com/opencv/opencv/tree/master/data/haarcascades

- Copy the xml files provided on BB to the same directory as your script.
- See the OpenCV documentation to refine the algorithm's params and reduce false positives.
- Provide the video path to VideoCapture to process videos.
"""

import cv2 as cv
import numpy as np

# print(cv.__version__)

# models' settings. XML files are located in the same directory as the script.
face_detector = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_detector = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")
mouth_detector = cv.CascadeClassifier("haarcascades/haarcascade_mcs_mouth.xml")

fps = 30
sleep = int(1000 / fps)

new_image_width = 640  # downsample the image

# camera object: 0 WebCam, or video path
cap = cv.VideoCapture("long_videos/video3.mov")  # cap = cv.VideoCapture(video_path)

# run continuously
while cap.isOpened():
    ret, frm = cap.read()
    if not ret:
        break

    # resize image to 640 width
    new_image_height = int(new_image_width * frm.shape[0] / frm.shape[1])
    frm_resized = cv.resize(frm, (new_image_width, new_image_height))

    # convert colour image to greyscale
    frm_grey = cv.cvtColor(frm_resized, cv.COLOR_BGR2GRAY)

    # apply face detection
    # can refine params (see doc https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)
    face_bbox = face_detector.detectMultiScale(
        frm_grey, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
    )
    if len(face_bbox) > 0:
        for face_x1, face_y1, face_width, face_height in face_bbox:
            face_RoI = frm_grey[
                face_y1 : face_y1 + face_height, face_x1 : face_x1 + face_width
            ]

            face_RoI_top_limit = int(
                face_height / 2
            )  # eye search region, from top to height/2
            face_RoI_bttm_limit = int(
                face_height * 0.55
            )  # mouth search region, bottom 45% of the face region

            # show top and bottom face sections
            cv.imshow(
                "top-bttm",
                np.vstack(
                    (
                        face_RoI[:face_RoI_top_limit, :],
                        face_RoI[face_RoI_bttm_limit:, :],
                    )
                ),
            )

            # apply eye detection
            eye_bboxes = eye_detector.detectMultiScale(
                face_RoI[:face_RoI_top_limit, :],
                scaleFactor=1.15,
                minSize=(15, 15),
                minNeighbors=10,
            )

            # apply mouth/smile detection to the face ROI
            smile_bboxes = mouth_detector.detectMultiScale(
                face_RoI[face_RoI_bttm_limit:, :],
                scaleFactor=1.15,
                minNeighbors=15,
                minSize=(15, 15),
            )

            # draw eye bounding boxes
            for eye_x1, eye_y1, eye_width, eye_height in eye_bboxes:
                p1 = (face_x1 + eye_x1, face_y1 + eye_y1)
                p2 = (face_x1 + eye_x1 + eye_width, face_y1 + eye_y1 + eye_height)
                cv.rectangle(frm_resized, p1, p2, (255, 0, 0), 2)

            # draw smile bounding boxes
            for mouth_x1, mouth_y1, mouth_width, mouth_height in smile_bboxes:
                p1 = (face_x1 + mouth_x1, face_y1 + face_RoI_bttm_limit + mouth_y1)
                p2 = (
                    face_x1 + mouth_x1 + mouth_width,
                    face_y1 + face_RoI_bttm_limit + mouth_y1 + mouth_height,
                )
                cv.rectangle(frm_resized, p1, p2, (0, 0, 255), 2)

            # draw face bounding box on the frame
            p1 = (face_x1, face_y1)
            p2 = (face_x1 + face_width, face_y1 + face_height)
            cv.rectangle(frm_resized, p1, p2, (0, 255, 0), 2)

    cv.imshow("frm_resized", frm_resized)
    key = cv.waitKey(sleep)
    if key == ord("q"):
        break

cv.destroyAllWindows()
