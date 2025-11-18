import random
import time

# Script to print shuffle and print names for recording

NAMES = [name.strip() for name in open("NAMES.txt").readlines()]
random.shuffle(NAMES)
for i in range(3,0,-1):
    print(i)
    time.sleep(1)

print('START!')
for n in NAMES:
    # next_time = time.time() + 3
    print(f"{i:2}/20 {n}")
    time.sleep(3)
    i += 1
