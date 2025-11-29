# checks that all files in short_videos are named correctly
from pathlib import Path
import glob
import re

clean = True
NAMES = [name.strip().lower() for name in open("NAMES.txt").readlines()]
for file in sorted(glob.glob("short_videos/*/*.mov"), reverse=True):
    if not re.match(r"short_videos/video\d{1,2}/[a-z]+\d{3}\.mov", file):
        print(Path(file).stem)
        clean = False

if clean:
    print("all files are correct")
