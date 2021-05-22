import cv2
from pathlib2 import Path
import time
path = Path(".")

path = path.glob("*.jpg")

# images = []

start = time.time()

for imagepath in path:
    img = cv2.imread(str(imagepath))
#     imS = cv2.resize(img, (940, 600))
#     images.append(img)
    median = cv2.medianBlur(img, 3)
    cv2.imshow('Gaussian Blurring', median)
    cv2.waitKey(0)
