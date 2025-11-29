import random
import time
from pathlib import Path

# Script to print shuffle and print names for recording

NAMES = [name.strip() for name in open("NAMES.txt").readlines()]
random.shuffle(NAMES)
i = 0
while Path(f"names{i:03}.txt").exists():
    i += 1
outfile = open(f"names{i:03}.txt", "x")
print(f"writing to names{i:03}.txt")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("START!")

start_time = time.time()
# next_time = start_time
for n in NAMES:
    next_time = start_time + float(3 * i)
    while time.time() < next_time:
        pass
    print(f"{i:2}/20 {n} ({int(time.time()) %100})")
    outfile.write(n + "\n")
    time.sleep(3)
    i += 1
outfile.close()
