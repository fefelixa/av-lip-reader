from pathlib import Path
import cv2
import os
import re
import glob
import random
from opt_einsum.path_random import random_choices
import sounddevice as sd
import soundfile as sf

# NOTE: VIDEO COLLECITON CANNOT BE AUTOMATED IN PYTHON
# THIS CODE DOES NOT WORK PROPERLY
# DO NOT USE

cam = cv2.VideoCapture(1)
FRAMEW = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAMEH = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = int(cam.get(cv2.CAP_PROP_FPS))
cam.set(cv2.CAP_PROP_FPS, 30.0)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
print(f"video format: {FRAMEW} x {FRAMEH} @ {FPS}fps")

NAMES = [name.strip() for name in open("NAMES.txt").readlines()]
DIR = "./videos"
names_dict = dict.fromkeys(NAMES, 0)
for name in NAMES:
    vids = glob.glob(f"videos/{name.lower()}*.mp4")
    names_dict[name] = len(vids)


SECONDS = 3
running = True
while running:
    name = random.choice(NAMES)
    # filename = f"videos/{name.lower()}{names_dict[name]+1:03}"  # videos/name001.mp4
    filename = "videos/test"
    vidfile = cv2.VideoWriter(filename + ".mp4", fourcc, FPS, (FRAMEW, FRAMEH))
    frame_counter = 0
    recording = True
    audio = sd.rec(SECONDS * 16000, samplerate=16000, channels=1)
    while recording:

        ret, frame = cam.read()
        vidfile.write(frame)
        cv2.imshow(name, frame)
        frame_counter += 1

        # if cv2.waitKey(1) == ord("q"):
        #     recording = False
        if frame_counter >= FPS * SECONDS:
            recording = False
    vidfile.release()
    cam.release()
    cv2.destroyAllWindows()
    norm_audio = 0.99 * audio / max(abs(audio))
    sf.write(filename + ".wav", norm_audio, 16000)
    names_dict[name] += 1
    running = False

print("finished!")
