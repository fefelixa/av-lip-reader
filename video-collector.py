import cv2

cam = cv2.VideoCapture(0)
FRAMEW = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAMEH = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cam.get(cv2.CAP_PROP_FPS))
print(f"video: {FRAMEW} x {FRAMEH} @ {FPS}fps")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (FRAMEW, FRAMEH))

recording = True
while recording:
    ret, frame = cam.read()
    out.write(frame)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        recording = False

cam.release()
out.release()
cv2.destroyAllWindows()
print('finished!')