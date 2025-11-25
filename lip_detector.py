# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
import cv2

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
smile_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")


frame = cv2.imread("frame.png")
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.equalizeHist(frame_gray)

faces = face_cascade.detectMultiScale(frame_gray)
for x1, y1, w1, h1 in faces:
    frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0,0), 4) #faces are blue

    face_roi = frame_gray[y1 : y1 + h1, x1 : x1 + w1]
    cv2.imshow("face", face_roi)
    key = cv2.waitKey(0)
    smile = smile_cascade.detectMultiScale(face_roi)
    i = 0
    for x2, y2, w2, h2 in smile:
        clr = i*255/len(smile)
        cv2.imshow("face", face_roi[y2:y2+h2,x2:x2+h2])
        key = cv2.waitKey(0)
        frame = cv2.rectangle(
            frame, (x1 + x2, y1 + y2), (x1 + x2 + w2, y1 + y2 + h2), (0, 255-clr, clr), 4 #eyes from green -> red
        )
        i+=1
cv2.imshow("face detection", frame)
k = cv2.waitKey(0)
