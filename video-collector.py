import cv2

cam = cv2.VideoCapture(0)

FRAMEW = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAMEH = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cam.get(cv2.CAP_PROP_FPS))
cam.set(cv2.CAP_PROP_FPS, 30.0)
print(f"video: {FRAMEW} x {FRAMEH} @ {FPS}fps")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, FPS, (FRAMEW, FRAMEH))

frame_counter = 0

SECONDS = 10

recording = True
while recording:

    ret, frame = cam.read()
    out.write(frame)
    cv2.imshow("Camera", frame)
    frame_counter += 1

    # if cv2.waitKey(1) == ord("q"):
    #     recording = False
    if frame_counter == FPS * SECONDS:
        recording = False

cam.release()
out.release()
cv2.destroyAllWindows()
print('finished!')