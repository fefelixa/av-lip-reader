import ffmpeg, os
import subprocess
from pathlib import Path

DIR = Path("long_videos/")
for filename in os.listdir(DIR):
    print(DIR / filename)
    if filename[-4:] == ".mov":
        try:
            os.mkdir(DIR/filename[:-4])
        except:
            pass
        # for i in range(20):
        #     outfile = filename[:-4] + f"_{i:03}.mov"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            DIR / filename,
            "-acodec",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            "3",
            "-vcodec",
            "copy",
            "-reset_timestamps",
            "1",
            "-map",
            "0",
            DIR / filename[:-4] / "%d.mov",
        ]
        subprocess.run(command)
        # input()
