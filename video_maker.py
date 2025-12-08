import random
import time
from pathlib import Path
import os
import subprocess
from tracemalloc import start
import librosa
from tqdm import tqdm

# Script to shuffle names and split recorded video

NAMES = [name.strip() for name in open("NAMES.txt").readlines()]
random.shuffle(NAMES)
i = 0
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("START!")

start_time = time.time()
next_time = start_time
for n in NAMES:
    print(f"{i:2}/20 {n}")
    next_time += 3.0
    i += 1
    while time.time() < next_time:
        pass


vidnum = int(input("video number").strip())
os.makedirs(f"short_videos/video{vidnum}/", exist_ok=True)

for i in tqdm(range(20), unit="vid"):
    cmd = [
        "ffmpeg",
        "-ss",
        str(i * 3),
        "-i",
        f"long_videos/video{vidnum}.mov",
        "-t",
        "3",
        "-c",
        "copy",
        f"short_videos/video{vidnum}/{NAMES[i].lower()}{vidnum:03}.mov",
        "-y",
        "-loglevel",
        "16",
    ]
    subprocess.run(cmd)
