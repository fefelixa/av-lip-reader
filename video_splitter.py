import os
import subprocess
import librosa
from pathlib import Path

DIR = Path("long_videos/")

filename = "video2.mov"
try:
    os.mkdir(DIR / filename[:-4])
except:
    pass
# for i in range(20):
#     outfile = filename[:-4] + f"_{i:03}.mov"
for filename in os.listdir(DIR):
    if filename.endswith('.mov'):
        try:
            os.mkdir(DIR / filename[:-4])
        except:
            pass
        for i in range(20):
            command2 = [
                "ffmpeg",
                "-ss",
                str(i * 3),
                "-i",
                DIR / filename,
                "-t",
                "3",
                "-c",
                "copy",
                DIR / filename[:-4]/f'{filename[:-4]}_{i:03}.mov',
                '-y'
            ]
            subprocess.run(command2)
# input()
